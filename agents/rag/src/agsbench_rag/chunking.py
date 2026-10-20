from fast_service import RequestContext
from typing import List
from .shared import agsbench_rag_fsm, Document
from .shared import DocSplittingInput, DocSplittingOutput
from .utils import get_splitter, get_lock, lru_cache, get_logger
from .settings import get_settings
from .zmq_utils import ZMQClient


class ChunkingEngine:
    def __init__(self):
        pass

    def execute(
        self,
        splitter_model: str,
        chunk_size: int,
        chunk_overlap: int,
        docs: List[Document],
        request_id: str = None,
    ) -> List[Document]:
        get_logger().debug(f"ChunkingEngine.execute start: {request_id}")
        text_splitter = get_splitter(
            splitter_model=splitter_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(docs)
        get_logger().debug(f"ChunkingEngine.execute end: {request_id}")
        return chunks


def chunking_server(addr):
    import zmq

    engine = ChunkingEngine()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(addr)

    while True:
        message = socket.recv_json()
        item: DocSplittingInput = DocSplittingInput.model_construct(
            **dict(message["item"])
        )
        docs = engine.execute(
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
            docs=[Document.model_construct(**docstr) for docstr in item.docs],
            request_id=message.get("request_id", None),
        )
        socket.send_json([doc.model_dump() for doc in docs])


class ChunkingClient:
    def __init__(self, engine_type: str, num_engines: int, lock_name: str = None):
        self.engine_type = engine_type
        self.lock_name = lock_name

        self.engine = None
        self.zmq_client = None
        if self.engine_type is None or self.engine_type in ["local", "local-lock"]:
            self.engine = ChunkingEngine()
        elif self.engine_type in ["zmq-thread", "zmq-process"]:
            addrs = [f"tcp://127.0.0.1:{31000 + i}" for i in range(num_engines)]
            self.zmq_client = ZMQClient(
                chunking_server, addrs, executor=self.engine_type.split("-")[1]
            )
        else:
            raise NotImplementedError(f"Unsupported loading engine: {self.engine_type}")

    def zmq_execute(
        self,
        item: DocSplittingInput,
        context: RequestContext = None,
    ) -> list[Document]:
        response = self.zmq_client.send_recv_json(
            {
                "item": item.model_dump(),
                "request_id": context.request_id,
            }
        )
        return [Document.model_construct(**dict(dstr)) for dstr in response]

    def local_execute(
        self, item: DocSplittingInput, context: RequestContext = None
    ) -> list[Document]:
        return self.engine.execute(
            splitter_model=item.splitter_model,
            chunk_size=item.chunk_size,
            chunk_overlap=item.chunk_overlap,
            docs=item.docs,
            request_id=context.request_id,
        )

    def execute(
        self, item: DocSplittingInput, context: RequestContext = None
    ) -> list[Document]:
        if self.engine is not None:
            if self.lock_name is not None:
                with get_lock(self.lock_name):
                    return self.local_execute(
                        item=item,
                        context=context,
                    )
            else:
                return self.local_execute(
                    item=item,
                    context=context,
                )
        elif self.zmq_client is not None:
            return self.zmq_execute(item=item, context=context)
        else:
            raise NotImplementedError(f"Unsupported loading engine: {self.engine_type}")


@lru_cache
def get_chunking_client():
    engine_type = get_settings().chunking_engine
    num_engines = get_settings().num_chunking_engines
    lock_name = get_settings().chunking_lock_name
    return ChunkingClient(
        engine_type=engine_type, num_engines=num_engines, lock_name=lock_name
    )


@agsbench_rag_fsm.fast_service
def split_docs(
    item: DocSplittingInput, context: RequestContext = None
) -> DocSplittingOutput:
    client = get_chunking_client()
    item.splitter_model = get_settings().splitter_model
    item.chunk_size = get_settings().chunk_size
    item.chunk_overlap = get_settings().chunk_overlap
    splits = client.execute(item=item, context=context)
    return DocSplittingOutput(docs=splits)
