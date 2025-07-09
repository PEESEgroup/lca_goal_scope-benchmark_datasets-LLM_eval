from transformers import Pipeline, pipeline
from typing import List, Optional, Tuple
import torch
from ragatouille import RAGPretrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, HF_ColBERT
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS

def answer_with_rag(
        question: str,
        llm: Pipeline,
        knowledge_index: FAISS,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    # configure reading LLM
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
    reading_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    rerank_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    RERANKER = HF_ColBERT.from_pretrained("colbert-ir/colbertv2.0")
    READER_LLM = pipeline(
        model=model,
        tokenizer=reading_tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    # configure orchestration rag prompt
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = reading_tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs


if __name__ == "__main__":
    KNOWLEDGE_VECTOR_DATABASE = new_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
    answer_with_rag("/vectorstore/vs_journal")