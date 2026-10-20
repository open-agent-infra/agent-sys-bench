"""QA scenario data preparation — wiki_qa dataset.

Downloads microsoft/wiki_qa (pinned HF revision) and generates:
  - docs/<document_id>.html — one local HTML document per Wikipedia page,
    assembled from the page's summary sentences shipped with wiki_qa
    (the dataset's candidate-answer pool), so no network access is needed
    at request time.
  - natural_questions.txt — tab-separated manifest for the RAG agent:
    question_id, question, document_id, document_title, doc path
    (relative to the output dir; the loadgen prefixes --doc_root).
"""

import html
import os
import pandas as pd
from datasets import load_dataset

# Pinned HF dataset revision for reproducibility.
HF_DATASET = "microsoft/wiki_qa"
HF_REVISION = "3f104672b5de699878fe7907afc486f0de325eb5"
MANIFEST_NAME = "natural_questions.txt"
DOCS_DIR = "docs"


def extra_args(parser):
    parser.add_argument("--nrows", type=int, default=0,
                        help="Number of questions to process (0 = all).")


def _write_doc(path, title, sentences):
    body = "\n".join(f"<p>{html.escape(s)}</p>" for s in sentences)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><title>{html.escape(title)}</title></head>"
                f"<body><h1>{html.escape(title)}</h1>\n{body}\n</body></html>\n")


def run(args):
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, DOCS_DIR), exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    print(f"Loading {HF_DATASET} (revision {HF_REVISION[:8]})...")
    ds = load_dataset(HF_DATASET, revision=HF_REVISION)
    df = pd.concat([ds[split].to_pandas() for split in ("train", "validation", "test")])

    questions = df[["question_id", "question", "document_title"]].drop_duplicates()
    if args.nrows > 0:
        questions = questions.head(args.nrows)

    # One document per page title; sentences keep their dataset order.
    titles = questions["document_title"].unique()
    sentences_by_title = df.groupby("document_title", sort=False)["answer"].agg(
        lambda s: s.drop_duplicates().tolist())
    doc_ids = {}
    for i, title in enumerate(titles):
        doc_id = f"D{i}"
        sentences = sentences_by_title[title]
        _write_doc(os.path.join(output_dir, DOCS_DIR, f"{doc_id}.html"), title, sentences)
        doc_ids[title] = doc_id

    with open(manifest_path, "w", encoding="utf-8") as f:
        for _, row in questions.iterrows():
            doc_id = doc_ids[row["document_title"]]
            f.write(f"{row['question_id']}\t{row['question']}\t{doc_id}\t"
                    f"{row['document_title']}\t{DOCS_DIR}/{doc_id}.html\n")
    print(f"Wrote {len(questions)} questions and {len(titles)} documents to {output_dir}")
