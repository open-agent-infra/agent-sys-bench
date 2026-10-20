from abc import abstractmethod
from typing import Union, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunkingFunction:
    @abstractmethod
    def __call__(self, documents: Union[str, List[str]]) -> List[str]:
        """ """

    @abstractmethod
    def chunk(self, documents: Union[str, List[str]]) -> List[str]:
        """ """


class NaiveChunkingFunction(BaseChunkingFunction):
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 128) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: Union[str, List[str]]) -> List[str]:
        if isinstance(documents, str):
            documents = [documents]
        docs = self.text_splitter.create_documents(documents)
        chunks = [doc.page_content for doc in docs]
        return chunks
