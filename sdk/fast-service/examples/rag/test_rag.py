import requests
import os
import json
from typing import BinaryIO

from fast_service import FastServiceConfig, FastServiceFile, RequestContext

from config import get_settings
from naive_rag import fast_service_manager
from naive_rag import indexing, DocIndexingOutput
from naive_rag import rag_chat, RAGChatInput, RAGChatOutput
from naive_rag import undo_indexing, UndoIndexingInput, UndoIndexingOutput
from naive_rag import drop_collection, DropCollectionInput, DropCollectionOutput
from naive_rag import rag_e2e, RAGE2EOutput


def indexing_requests(
    collection_name: str,
    files: list[BinaryIO],
) -> DocIndexingOutput:
    y = requests.post(
        "http://127.0.0.1:7090/naive_rag.indexing",
        data={
            "params": json.dumps(
                {
                    "collection_name": collection_name,
                    "context": RequestContext(
                        request_id=RequestContext.generate_request_id(),
                        call_id=RequestContext.generate_call_id(),
                    ).model_dump(),
                }
            )
        },
        files=[("files", (os.path.basename(file.name), file)) for file in files],
    )
    return DocIndexingOutput(**dict(y.json()))


def rag_chat_requests(item: RAGChatInput) -> RAGChatOutput:
    y = requests.post(
        "http://127.0.0.1:7090/naive_rag.rag_chat",
        json={"item": item.model_dump()},
    )
    return RAGChatOutput(**dict(y.json()))


def undo_indexing_requests(item: UndoIndexingInput) -> UndoIndexingOutput:
    y = requests.post(
        "http://127.0.0.1:7090/naive_rag.undo_indexing",
        json={"item": item.model_dump()},
    )
    return UndoIndexingOutput(**dict(y.json()))


def drop_collection_requests(item: DropCollectionInput) -> DropCollectionOutput:
    y = requests.post(
        "http://127.0.0.1:7090/naive_rag.drop_collection",
        json={"item": item.model_dump()},
    )
    return DropCollectionOutput(**dict(y.json()))


def rag_e2e_requests(
    collection_name: str,
    files: list[BinaryIO],
    question: str,
    context_limit: int = 3,
) -> RAGE2EOutput:
    y = requests.post(
        "http://127.0.0.1:7090/naive_rag.rag_e2e",
        data={
            "params": json.dumps(
                {
                    "collection_name": collection_name,
                    "question": question,
                    "context_limit": context_limit,
                    "context": RequestContext(
                        request_id=RequestContext.generate_request_id(),
                        call_id=RequestContext.generate_call_id(),
                    ).model_dump(),
                }
            )
        },
        files=[("files", (os.path.basename(file.name), file)) for file in files],
    )
    return RAGE2EOutput(**dict(y.json()))


def test_requests_interface(file_paths, collection_name, question, questions):
    files = [open(file_path, "rb") for file_path in file_paths]
    y1 = indexing_requests(collection_name=collection_name, files=files)
    print(f"y1 => {y1.model_dump()}")
    y2 = rag_chat_requests(
        RAGChatInput(
            history=[],
            question=question,
            collection_name=collection_name,
            context_limit=3,
        )
    )
    print(f"y2 => {y2.model_dump()}")

    y3 = undo_indexing_requests(
        UndoIndexingInput(
            collection_name=collection_name,
            filenames=[os.path.basename(file.name) for file in files],
        )
    )
    print(f"y3 => {y3.model_dump()}")

    y4 = drop_collection_requests(DropCollectionInput(collection_name=collection_name))
    print(f"y4 => {y4.model_dump()}")

    files = [open(file_path, "rb") for file_path in file_paths]
    y5 = rag_e2e_requests(
        collection_name=collection_name,
        files=files,
        question=question,
        context_limit=3,
    )
    print(f"y5 => {y5.model_dump()}")


def test_client_interface(file_paths, collection_name, question, questions):
    config = FastServiceConfig.load_from_file("./config/naive-rag-client.yml")
    fast_service_manager.setup_client_mode(config)

    files = [FastServiceFile.load(open(file_path, "rb")) for file_path in file_paths]
    y1 = indexing(collection_name=collection_name, files=files)
    print(f"y1 => {y1.model_dump()}")
    y2 = rag_chat(
        RAGChatInput(
            history=[],
            question=question,
            collection_name=collection_name,
            context_limit=3,
        )
    )
    print(f"y2 => {y2.model_dump()}")

    answers = []
    for i, q in enumerate(questions):
        item = RAGChatInput(
            history=[], question=q, collection_name=collection_name, context_limit=3
        )
        answer = rag_chat(item=item).model_dump()
        print(f"q-{i} => {answer}")
        answers.append(answer)
    print(f"answers: {answers}")

    y3 = undo_indexing(
        UndoIndexingInput(
            collection_name=collection_name,
            filenames=[file.get_filename() for file in files],
        )
    )
    print(f"y3 => {y3.model_dump()}")
    y4 = drop_collection(DropCollectionInput(collection_name=collection_name))
    print(f"y4 => {y4.model_dump()}")
    files = [FastServiceFile.load(open(file_path, "rb")) for file_path in file_paths]
    y5 = rag_e2e(
        collection_name=collection_name, files=files, question=question, context_limit=3
    )
    print(f"y5 => {y5.model_dump()}")


if __name__ == "__main__":
    file_dir = "./.cache"
    file_paths = [
        os.path.join(file_dir, "example_1.pdf"),
        os.path.join(file_dir, "example_2.pdf"),
    ]
    collection_name = get_settings().vdb_collection
    question = "Please summarize the papers for me."
    questions = [
        f"Please help compare the {part} of these papers."
        for part in [
            "Introduction",
            "Background",
            "Solution",
            "Evaluation",
            "Related Work",
        ]
    ]

    # test requests interface
    test_requests_interface(file_paths, collection_name, question, questions)

    # test client mode interface
    test_client_interface(file_paths, collection_name, question, questions)
