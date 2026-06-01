from transformers import Pipeline, pipeline
from typing import Any
import torch
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import constants
import os


def answer_with_rag(
        system_description: str,
        question: str,
        hestia: str,
        llm: Pipeline,
        reading_tokenizer: AutoTokenizer,
        knowledge_index: FAISS,
        num_retrieved_docs: int = 20, # vary between 10, 20, 30
        num_docs_final: int = 3, # vary between 1, 3, 5
) -> tuple[Any, list[str]]:
    """
    method to answer a given query using RAG data
    """
    # gather documents with retriever first so we can build the context payload
    print(f"=> Retrieving documents for query {question}...")
    relevant_docs = knowledge_index.similarity_search(query=system_description+ " " + question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # reranking documents
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # pair the query with each document and calculate scores
    pairs = [[system_description + " " + question, doc] for doc in relevant_docs]
    scores = rerank_model.predict(pairs)

    # rank scores
    scored_docs = sorted(zip(scores, relevant_docs), key=lambda x: x[0], reverse=True)

    counter = 0
    reranked_docs = []
    for score, doc in scored_docs:
        counter = counter + 1
        if counter <= num_docs_final:
            reranked_docs.append(doc)

    # build the context block
    context = "\nAdditional Context:\n"
    context += "".join([f"Source {str(i)}:::\n" + doc for i, doc in enumerate(reranked_docs)])

    # configure the prompt and insert string variables
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """You are an expert on agricultural life cycle assessment (LCA). 
            Please summarize the life cycle assessment information that is relevant to the description of the system, 
            life cycle assessment sub-task question and context using the HESTIA schema information. 
            Please use as few words as necessary. 
            You do not need to provide document numbers or restate parts of the prompt.""",
        },
        {
            "role": "user",
            "content": f"""
            Description of the System: {system_description}
            ---
            Question: {question}
            ---
            HESTIA schema information: {hestia}
            ---
            Context: {context}"""
        },
    ]

    # convert chat structure into the raw model-specific tokens string
    final_prompt = reading_tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    # retrieve an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    # String processing to isolate the assistant response from the prompt wrapper
    generated_answer = answer.split("<|start_header_id|>assistant<|end_header_id|>")[1]
    generated_answer = generated_answer.strip()
    print(f"=> model answers \"{generated_answer}\"\n\n")
    
    return generated_answer, relevant_docs


def model_config(model_name="nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4"):
    """
    set up LLM model config for summarizing RAG results
    :param model_name: name of the model - in our case Llama3.2-3B-Instruct
    :return: the LLM pipeline and the tokenizer
    """
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Manually map the Llama chat template directly to the tokenizer config
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )
    
    # initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        max_new_tokens=256, # do not need a lot of information here
        do_sample=False
    )

    return pipe, tokenizer


if __name__ == "__main__":
    embeddings = constants.EMBED_MODEL
    os.chdir('llm-goal-scope')
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")
    reader, tokenizer = model_config()
    print("model configured")
    # test question to make sure things are working
    question = "what is a functional unit for sheep production in the UK?"
    answer, docs = answer_with_rag("", question, "", reader, tokenizer, vdb)
    print(answer)
