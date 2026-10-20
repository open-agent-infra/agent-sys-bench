from typing import List
from abc import abstractmethod
import numpy as np
import onnxruntime
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


class BaseEmbeddingFunction:
    @abstractmethod
    def __call__(self, texts: List[str]):
        """ """

    @abstractmethod
    def encode(self, data: List[str]):
        """ """


class OnnxEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "GPTCache/paraphrase-albert-onnx",
        tokenizer_name: str = "GPTCache/paraphrase-albert-small-v2",
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model_name = model_name
        onnx_model_path = hf_hub_download(repo_id=model_name, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

        if device == "cpu":
            self.ort_session.set_providers(["CPUExecutionProvider"])
        else:
            self.ort_session.set_providers(["CUDAExecutionProvider"])

        config = AutoConfig.from_pretrained(tokenizer_name)
        self.__dimension = config.hidden_size

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def encode(self, data: List[str]) -> List[np.array]:
        return self._encode(data)

    def _encode(self, texts: List[str]) -> List[np.array]:
        return [self._to_embedding(text) for text in texts]

    def _to_embedding(self, data: str, **_):
        encoded_text = self.tokenizer.encode_plus(data, padding="max_length")

        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"])
            .astype("int64")
            .reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"])
            .astype("int64")
            .reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"])
            .astype("int64")
            .reshape(1, -1),
        }

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self._post_proc(ort_feat, ort_inputs["attention_mask"])
        emb = emb.flatten()
        return emb / np.linalg.norm(emb)

    def _post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dim(self):
        return self.__dimension


class SentenceTransformerEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        query_instruction: str = "",
        doc_instruction: str = "",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        _model_config = dict(
            {"model_name_or_path": model_name, "device": device}, **kwargs
        )
        self.model = SentenceTransformer(**_model_config)

    def __call__(self, texts: List[str]) -> List[np.array]:
        return self._encode(texts)

    def _encode(self, texts: List[str]) -> List[np.array]:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return list(embs)

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def encode(self, data: List[str]):
        return self._encode(data)

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        instructed_queries = [self.query_instruction + query for query in queries]
        return self._encode(instructed_queries)

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        instructed_documents = [
            self.doc_instruction + document for document in documents
        ]
        return self._encode(instructed_documents)
