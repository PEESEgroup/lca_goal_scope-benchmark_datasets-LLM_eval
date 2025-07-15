from transformers import Pipeline, pipeline
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
import constants
import pprint


def answer_with_rag(
        question: str,
        llm: Pipeline,
        reading_tokenizer: AutoTokenizer,
        knowledge_index: FAISS,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
) -> tuple[Any, list[str]]:
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
    relevant_docs = relevant_docs[:num_docs_final]

    # TODO: reranking documents

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs


def get_context(question, knowledge_index, num_retrieved_docs=5):
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text
    return relevant_docs


def model_config(model_name="HuggingFaceH4/zephyr-7b-beta"):
    # configure reading LLM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # TODO: remove quantization???
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    reading_tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    return READER_LLM, reading_tokenizer


if __name__ == "__main__":
    embeddings = constants.EMBED_MODEL
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")
    reader, tokenizer = model_config()
    print("model configured")
    question = "what is a functional unit for milk?"
    answer, docs = answer_with_rag(question, reader, tokenizer, vdb)
    pprint.pprint(answer, indent=2, width=120, depth=2)
    pprint.pprint(docs, indent=2, width=120, depth=2)
