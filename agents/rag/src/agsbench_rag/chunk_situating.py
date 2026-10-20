from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
from fast_service import RequestContext
from .shared import agsbench_rag_fsm
from .shared import Document
from .shared import SituatingChunksInput, SituatingChunksOutput
from .prompt_templates import ALL_DEFAULT_TEMPLATES
from .utils import get_llm, get_logger
from .settings import SituatingLLMSettings, get_settings


@agsbench_rag_fsm.fast_service
def situate_chunk(
    doc: Document, chunk: Document, context: RequestContext = None
) -> Document:
    settings = SituatingLLMSettings.from_default_settings()
    llm = get_llm(settings=settings)
    prompt_template = ALL_DEFAULT_TEMPLATES["contextual_retrieval"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm_input = prompt.invoke(
        input={"doc_content": doc.page_content, "chunk_content": chunk.page_content}
    )
    get_logger().debug(f"situate_chunk input {context.request_id}: {llm_input}")
    try:
        llm_output = llm.invoke(
            input=llm_input,
            extra_headers={"X-Request-Id": context.call_id} if context else None,
        )
        answer = f"{chunk.page_content}\n\n{llm_output.content}"
        get_logger().debug(f"situate_chunk output {context.request_id}: {llm_output}")
    except Exception as e:
        answer = chunk.page_content
        get_logger().error(f"situate_chunk Error invoking LLM: {e}")
    return Document(page_content=answer, metadata=chunk.metadata)


@agsbench_rag_fsm.fast_service
def situate_chunks(
    item: SituatingChunksInput, context: RequestContext = None
) -> SituatingChunksOutput:
    batch_size = get_settings().situating_batch_size
    parallel_situating = get_settings().parallel_situating
    new_chunks = []
    if parallel_situating:
        if batch_size == 0:
            batch_size = len(item.chunks)
        for i in range(0, len(item.chunks), batch_size):
            batch_chunks = item.chunks[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(
                        situate_chunk,
                        doc=item.docs[chunk.metadata.get("doc_id", 0)],
                        chunk=chunk,
                        context=context,
                    )
                    for chunk in batch_chunks
                ]
            new_chunks.extend([future.result() for future in futures])
    else:
        for chunk in item.chunks:
            new_chunk = situate_chunk(
                doc=item.docs[chunk.metadata.get("doc_id", 0)],
                chunk=chunk,
                context=context,
            )
            new_chunks.append(new_chunk)
    return SituatingChunksOutput(chunks=new_chunks)
