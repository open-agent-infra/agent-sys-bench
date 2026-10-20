# Example: Naive RAG

This project provides a typical agentic workflow known as naive RAG, which allows users to upload documents to the server and ask questions based on those documents.

### Notes:
1. Currently, the system only supports PDF files for document uploads, and users can upload only one file at a time.
2. The backend LLM engine must be vLLM deployed locally.

## Prerequisites

### 1. Install Required Packages

The required packages are listed in `examples/rag/requirements.txt`. You can install them by running the following command in the `example/rag` directory:

```bash
pip install -r requirements.txt
```

### 2. Launch vLLM for LLM Inference

``` bash
vllm serve google/gemma-2-2b-it
```

### 3. (Optional) Create a Dotenv File for Environment Variables

Create a file named `.env` in the `example/rag` directory to set up environment variables for the Naive RAG application. Here is an example configuration:

```bash
APP_NAME="Naive RAG"

VDB_URI="http://localhost:19530"
VDB_COLLECTION="test"

EMBEDDING_DEVICE="cuda"

RAW_FILES_DIR=".cache/files"

CHUNK_SIZE=256
CHUNK_OVERLAP=128

OPENAI_API_KEY="EMPTY"
OPENAI_BASE_URL="http://localhost:8000/v1"
```

If the `example/rag/.env` file does not exist, Naive RAG will use the default settings specified in `example/rag/config.py`.

## Launch Milvus Standalone
Ensure you are in the `example/rag` directory, then run the following command:
``` bash
python launch_milvus_standalone.py
```

## Launch Naive RAG

Ensure you are in the `example/rag` directory, then run the following command:

``` bash
python launch_rag.py --fast_service_config config/naive-rag.yml
```

## Test Naive RAG

Create directory `example/rag/.cache`, and put a pdf file named as `example.pdf` in this directory. (Alternative: you can also change the `file_path` in `test_rag.py` to specify the path to pdf file for testing.)

In another terminal, navigate to the `example/rag` directory and execute the following command:

``` bash
python test_rag.py
```

You will see something like the following:

``` txt
y1 => {'count': 635, 'cost': 0.0}
y2 => {'answer': 'The most related work to the paper "When ChatGPT Meets Smart" is **"Alternation" by Meyer (1981)**. \n\nHere\'s why:\n\n* **The context snippet provides a reference to a paper by Meyer in 1981 titled "Alternation"**. This paper likely deals with a concept of alternation which is a core idea in the paper "When ChatGPT Meets Smart".\n* **The phrase "when ChatGPT meets smart" suggests a field of research that involves combining AI and other technologies.**  \n* **Comparing AI models and technologies is a major theme explored in the field of AI**. \n\nTherefore, the paper "Alternation" by Meyer (1981) would be the most directly related work, as it deals with a fundamental concept that the paper "When ChatGPT Meets Smart" likely builds upon. \n', 'input_tokens': 180, 'output_tokens': 177}
...
```

## Test Naive RAG with Load Gen
We need to prepare a list of requests for the services we are going to test. 

First, let download several papers as the documents for QA. Let's save files into `.cache/download/papers` inside `naive_rag` directory.
``` bash
mkdir -p .cache/download/papers
cd .cache/download/papers
curl -o paper1.pdf https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
curl -o paper2.pdf https://www.usenix.org/system/files/osdi24-sun-biao.pdf
curl -o paper3.pdf https://www.usenix.org/system/files/osdi24-agrawal.pdf
curl -o paper4.pdf https://www.usenix.org/system/files/osdi24-fu.pdf
curl -o paper5.pdf https://www.usenix.org/system/files/osdi24-lee.pdf
```

Next, we need to generate a `requests.txt` as input for for load generator.

under `naive_rag/.cache/`, run the following script:
``` python
import os


dir_path = "./download/papers"
dir_path = os.path.abspath(dir_path)
file_paths = []
for file in os.listdir(dir_path):
    if file.endswith('.pdf'):
        file_path = os.path.join(dir_path, file)
        print(file_path)
        file_paths.append(file_path)

with open("requests.txt", "w") as file:
    for i, file_path in enumerate(file_paths):
        file.write(f"{i}, rag_e2e, {file_path}\n")
```

Finally, we can run load_gen.py directly for testing
``` python
python load_gen.py
```