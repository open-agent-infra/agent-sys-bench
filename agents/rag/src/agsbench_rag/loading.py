import os, json
from typing import Union, List
from fast_service import RequestContext
import urllib3
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyMuPDFLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from .settings import get_settings
from .shared import agsbench_rag_fsm
from .shared import DocLoadingInput, DocLoadingOutput
from .utils import get_logger, lru_cache, get_lock
from .zmq_utils import ZMQClient

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LoadingEngine:
    def __init__(self):
        pass

    def execute(
        self, type: Union[str, List[str]], uri: str, request_id: str = None
    ) -> list[Document]:
        get_logger().debug(f"LoadingEngine.load start for {request_id}: {type} {uri}")
        if type == "web":
            docs = self.load_web_docs(uri, request_id=request_id)
        elif type == "file":
            docs = self.load_pdf_docs(uri, request_id=request_id)
        elif type == "wiki":
            docs = self.load_wiki_docs(uri, request_id=request_id)
        elif type == "local_html":
            docs = self.load_local_html_docs(uri, request_id=request_id)
        else:
            raise NotImplementedError(f"Unsupported type: {type}")
        get_logger().debug(f"LoadingEngine.load end for {request_id}: {type} {uri}")
        return docs

    def load_web_docs(
        self, web_paths: Union[str, List[str]], request_id: str = None
    ) -> list[Document]:
        get_logger().debug(f"load_web_docs web_paths {request_id}: {web_paths}")
        if isinstance(web_paths, str):
            web_paths = [web_paths]
        if get_settings().loading_persistency is not None:
            # loading from persistency
            docs = []
            new_urls = []
            for url in web_paths:
                dirs = url.replace("://", "/").split("/")
                loading_dir = os.path.join(get_settings().loading_persistency, *dirs)
                if os.path.exists(loading_dir):
                    get_logger().debug(
                        f"Loading {url} from persistency {request_id}: {loading_dir}"
                    )
                    # each file is a document
                    _docs = []
                    loading_files = sorted(
                        [f for f in os.listdir(loading_dir) if f.endswith(".json")],
                        key=lambda x: int(x.split(".")[0]),
                    )
                    try:
                        for file in loading_files:
                            with open(
                                os.path.join(loading_dir, file), "r", encoding="utf-8"
                            ) as f:
                                doc = json.load(f)
                                _docs.append(Document.model_construct(**doc))
                    except Exception as e:
                        print(f"Error on {loading_dir}: {e}")
                    get_logger().debug(
                        f"Loaded {len(_docs)} docs from persistency {request_id}"
                    )
                    docs.extend(_docs)
                else:
                    new_urls.append(url)
            if new_urls:
                get_logger().debug(
                    f"Loading {len(new_urls)} new urls from web {request_id}: {new_urls}"
                )
                loader = WebBaseLoader(
                    web_path=new_urls,
                    header_template={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36"
                    },
                    continue_on_failure=True,
                    verify_ssl=False,
                    requests_kwargs={"allow_redirects": False, "timeout": 60},
                )
                _docs = loader.aload()
                get_logger().debug(f"Loaded {len(_docs)} docs from web {request_id}")
                docs.extend(_docs)

                # save to persistency
                for url in new_urls:
                    sub_docs = []
                    for doc in _docs:
                        if doc.metadata.get("source") == url:
                            sub_docs.append(doc)

                    dirs = url.replace("://", "/").split("/")
                    loading_dir = os.path.join(
                        get_settings().loading_persistency, *dirs
                    )
                    get_logger().debug(
                        f"Saving to persistency {request_id}: {loading_dir} with {len(sub_docs)} docs for {url}"
                    )
                    os.makedirs(loading_dir, exist_ok=True)

                    for i, doc in enumerate(sub_docs):
                        with open(
                            os.path.join(loading_dir, f"{i}.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
                    get_logger().debug(
                        f"Saved {len(sub_docs)} docs for {url} {request_id}"
                    )
            return docs
        else:
            get_logger().debug(f"Loading from web {request_id}: {web_paths}")
            loader = WebBaseLoader(
                web_paths=web_paths,
                header_template={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36"
                },
                continue_on_failure=True,
                verify_ssl=False,
                requests_kwargs={"allow_redirects": False, "timeout": 60},
            )
            docs = loader.aload()
            get_logger().debug(f"Loaded {len(docs)} docs from web {request_id}")
            return docs

    def load_pdf_docs(self, file_path: str, request_id: str = None) -> list[Document]:
        if get_settings().loading_persistency is not None:
            # loading from persistency
            dirs = file_path.replace("://", "/").split("/")
            loading_dir = os.path.join(get_settings().loading_persistency, *dirs)
            if os.path.exists(loading_dir):
                get_logger().debug(
                    f"Loading from persistency {request_id}: {loading_dir}"
                )
                # each file is a document
                docs = []
                for file in sorted(
                    os.listdir(loading_dir), key=lambda x: int(x.split(".")[0])
                ):
                    with open(
                        os.path.join(loading_dir, file), "r", encoding="utf-8"
                    ) as f:
                        doc = json.load(f)
                        docs.append(Document.model_construct(**doc))
                return docs
            else:
                loader = PyMuPDFLoader(file_path=file_path)
                docs = loader.load()

                # save to persistency
                get_logger().debug(f"Saving to persistency {request_id}: {loading_dir}")
                os.makedirs(loading_dir, exist_ok=True)
                for i, doc in enumerate(docs):
                    with open(
                        os.path.join(loading_dir, f"{i}.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
                return docs
        else:
            loader = PyMuPDFLoader(file_path=file_path)
            return loader.load()

    def load_wiki_docs(self, query: str, request_id: str = None) -> list[Document]:
        def _load_from_wiki(query):
            loader = WikipediaLoader(query=query)
            try:
                docs = loader.load()
            except Exception as e:
                get_logger().error(f"load_wiki_docs: Failed to load from wiki: {e}")
                docs = []
            return docs

        if get_settings().loading_persistency is not None:
            # loading from persistency
            dir_tag = query.replace(" ", "_")
            loading_dir = os.path.join(get_settings().loading_persistency, f"{dir_tag}")
            if os.path.exists(loading_dir):
                get_logger().debug(
                    f"Loading from persistency {request_id}: {loading_dir}"
                )
                # each file is a document
                docs = []
                for file in sorted(
                    os.listdir(loading_dir), key=lambda x: int(x.split(".")[0])
                ):
                    with open(
                        os.path.join(loading_dir, file), "r", encoding="utf-8"
                    ) as f:
                        doc = json.load(f)
                        docs.append(Document.model_construct(**doc))
                return docs
            else:
                docs = _load_from_wiki(query)

                # save to persistency
                get_logger().debug(f"Saving to persistency {request_id}: {loading_dir}")
                os.makedirs(loading_dir, exist_ok=True)
                for i, doc in enumerate(docs):
                    with open(
                        os.path.join(loading_dir, f"{i}.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
                return docs
        else:
            docs = _load_from_wiki(query)
            return docs

    def load_local_html_docs(
        self, file_path: str, request_id: str = None
    ) -> list[Document]:
        loader = UnstructuredHTMLLoader(file_path=file_path)
        docs = loader.load()
        return docs


def loading_server(addr: str = "tcp://127.0.0.1:30000"):
    import zmq

    engine = LoadingEngine()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(addr)

    while True:
        message = socket.recv_json()
        docs = engine.execute(
            type=message["type"],
            uri=message["uri"],
            request_id=message.get("request_id", None),
        )
        socket.send_json([doc.model_dump() for doc in docs])


class LoadingClient:
    def __init__(self, engine_type: str, num_engines: int, lock_name: str = None):
        self.engine_type = engine_type
        self.lock_name = lock_name

        self.engine = None
        self.zmq_client = None
        if self.engine_type is None or self.engine_type in ["local", "local-lock"]:
            self.engine = LoadingEngine()
        elif self.engine_type in ["zmq-thread", "zmq-process"]:
            addrs = [f"tcp://127.0.0.1:{30000 + i}" for i in range(num_engines)]
            self.zmq_client = ZMQClient(
                loading_server, addrs, executor=self.engine_type.split("-")[1]
            )
        else:
            raise NotImplementedError(f"Unsupported loading engine: {self.engine_type}")

    def zmq_execute(
        self, item: DocLoadingInput, request_id: str = None
    ) -> list[Document]:
        response = self.zmq_client.send_recv_json(
            {"type": item.type, "uri": item.uri, "request_id": request_id}
        )
        return [Document.model_construct(**dict(dstr)) for dstr in response]

    def local_execute(
        self, item: DocLoadingInput, request_id: str = None
    ) -> list[Document]:
        return self.engine.execute(type=item.type, uri=item.uri)

    def execute(self, item: DocLoadingInput, request_id: str = None) -> list[Document]:
        if self.engine is not None:
            if self.lock_name is not None:
                with get_lock(self.lock_name):
                    return self.local_execute(item=item, request_id=request_id)
            else:
                return self.local_execute(item=item, request_id=request_id)
        elif self.zmq_client is not None:
            return self.zmq_execute(item=item, request_id=request_id)
        else:
            raise NotImplementedError(f"Unsupported loading engine: {self.engine_type}")


@lru_cache
def get_loading_client() -> LoadingClient:
    engine_type = get_settings().loading_engine
    num_engines = get_settings().num_loading_engines
    lock_name = get_settings().loading_lock_name
    return LoadingClient(
        engine_type=engine_type, num_engines=num_engines, lock_name=lock_name
    )


@agsbench_rag_fsm.fast_service
def load_docs(
    item: DocLoadingInput, context: RequestContext = None
) -> DocLoadingOutput:
    client = get_loading_client()
    docs = client.execute(item, request_id=context.request_id)
    # sort docs by source and hash
    docs = sorted(
        docs, key=lambda x: (x.metadata.get("source", ""), hash(x.page_content))
    )
    return DocLoadingOutput(docs=docs)
