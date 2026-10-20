DEFAULT_MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""

DEFAULT_MULTI_QUERY_TEMPLATE_WITH_FUSION = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

DEFAULT_QUERY_DECOMPOSITION_TEMPLATE = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

DEFAULT_QUERY_STEP_BACK_TEMPLATE = """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:

Could the members of The Police perform lawful arrests? -> what can the members of The Police do?
Jan Sindel's was born in what country? -> what is Jan Sindel's personal history?

{question}"""

DEFAULT_HYDE_TEMPLATE = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

DEFAULT_QUERY_REWRITING_TEMPLATE = """ \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """

DEFAULT_DIRECT_GENERATION_TEMPLATE = """You are an assistant for question-answering tasks.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
\nQuestion: {question}
"""

DEFAULT_GENERATION_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
\nQuestion: {question}
\nContext: {context}
\nAnswer: """

DEFAULT_DECOMPOSITION_RECURSIVE_GENERATION_TEMPLATE = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question:

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

DEFAULT_DECOMPOSITION_INDIVIDUAL_GENERATION_TEMPLATE = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

DEFAULT_STEP_BACK_GENERATION_TEMPLATE = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {context}
# {step_back_context}

# Original Question: {question}
# Answer: """

DEFAULT_HYDE_GENERATION_TEMPLATE = """Answer the following question based on this context:

{context}

Question: {question}
"""

CONTEXTUAL_RETRIEVAL_PROMPT = """
    <document>
    {doc_content}
    </document>


    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>


    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

DEFAULT_RETRIEVAL_JUDGING_TEMPLATE = """You are an expert assistant that helps determine whether a question requires external knowledge retrieval or can be answered with general knowledge.

Question: {question}

Please analyze this question and determine if it requires specific factual information, recent data, or domain-specific knowledge that would benefit from document retrieval.

Answer with only "YES" if retrieval is needed, or "NO" if the question can be answered with general knowledge alone.

Answer:"""

# reference: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_cohere.ipynb
DEFAULT_DOC_RELEVANCE_JUDGING_TEMPLATE = """You are an expert assistant that evaluates whether a document is relevant to answering a specific question.

Question: {question}

Document: {context}

Please determine if this document contains information that is relevant and useful for answering the question.

Answer with only "YES" if the document is relevant, or only "NO" if it is not relevant.

Answer:"""

DEFAULT_HALLUCINATION_JUDGING_TEMPLATE = """You are an expert assistant that evaluates whether an answer is grounded in the provided context or contains hallucinated information.

Context: {context}

Answer: {answer}

Please determine if the answer is fully supported by the context provided. Look for any claims, facts, or details in the answer that are not present in or cannot be reasonably inferred from the context.

Answer with only "YES" if the answer is well-grounded (no hallucination), or only "NO" if the answer contains unsupported information (hallucination detected).

Answer:"""

DEFAULT_ANSWER_JUDGING_TEMPLATE = """You are an expert assistant that evaluates the quality of answers to questions.

Question: {question}

Answer: {answer}

Please evaluate if this answer adequately addresses the question. Consider:
- Does it directly answer what was asked?
- Is it complete and informative?
- Is it clear and well-structured?

Answer with only "YES" if the answer is of good quality, or only "NO" if it needs improvement.

Answer:"""

DEFAULT_COMPLETENESS_JUDGING_TEMPLATE = """Question: {question}
Answer: {answer}

Is this answer complete and comprehensive? Does it address all aspects of the question?
Answer with only "YES" or "NO"."""

DEFAULT_CLARITY_JUDGING_TEMPLATE = """Answer: {answer}

Is this answer clear, well-structured, and easy to understand?
Answer with only "YES" or "NO"."""

DEFAULT_CONTEXTUAL_QUERY_ENHANCEMENT_TEMPLATE = """Enhance the following query by incorporating the provided context to make it more specific and targeted.

Original query: {query}
{context_info}
Enhanced query:"""


ALL_DEFAULT_TEMPLATES = {
    "multi_query": DEFAULT_MULTI_QUERY_TEMPLATE,
    "multi_query_with_fusion": DEFAULT_MULTI_QUERY_TEMPLATE_WITH_FUSION,
    "query_decomposition": DEFAULT_QUERY_DECOMPOSITION_TEMPLATE,
    "query_step_back": DEFAULT_QUERY_STEP_BACK_TEMPLATE,
    "hyde": DEFAULT_HYDE_TEMPLATE,
    "query_rewriting": DEFAULT_QUERY_REWRITING_TEMPLATE,
    "direct_generation": DEFAULT_DIRECT_GENERATION_TEMPLATE,
    "normal_generation": DEFAULT_GENERATION_TEMPLATE,
    "decomposition_recursive_generation": DEFAULT_DECOMPOSITION_RECURSIVE_GENERATION_TEMPLATE,
    "decomposition_individual_generation": DEFAULT_DECOMPOSITION_INDIVIDUAL_GENERATION_TEMPLATE,
    "step_back_generation": DEFAULT_STEP_BACK_GENERATION_TEMPLATE,
    "hyde_generation": DEFAULT_HYDE_GENERATION_TEMPLATE,
    "contextual_retrieval": CONTEXTUAL_RETRIEVAL_PROMPT,
    "retrieval_judging": DEFAULT_RETRIEVAL_JUDGING_TEMPLATE,
    "doc_relevance_judging": DEFAULT_DOC_RELEVANCE_JUDGING_TEMPLATE,
    "hallucination_judging": DEFAULT_HALLUCINATION_JUDGING_TEMPLATE,
    "answer_judging": DEFAULT_ANSWER_JUDGING_TEMPLATE,
    "completeness_judging": DEFAULT_COMPLETENESS_JUDGING_TEMPLATE,
    "clarity_judging": DEFAULT_CLARITY_JUDGING_TEMPLATE,
    "contextual_query_enhancement": DEFAULT_CONTEXTUAL_QUERY_ENHANCEMENT_TEMPLATE,
}
