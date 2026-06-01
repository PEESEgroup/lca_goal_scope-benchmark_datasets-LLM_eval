import os
import sys

# Force the OS to prioritize Conda's modern C++ libraries
os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# might need to use the following to run on command line in AWS: 
# LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH /opt/conda/bin/python /home/sagemaker-user/llm-goal-scope/rag_retrieval.py

from vllm import LLM, SamplingParams
from transformers import Pipeline, pipeline, AutoTokenizer, AutoModelForCausalLM
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
        num_tokens: int = 256, # vary between 128/256/512
        temperature: float = 0.0 # 0.0, 0.33, 0.9
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
    sampling_params = SamplingParams(max_tokens=num_tokens, temperature=temperature)
    outputs = llm.generate([final_prompt], sampling_params)
    answer = outputs[0].outputs[0].text
    
    # vLLM outputs the newly generated text directly, 
    generated_answer = answer.strip()
    
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
    
    llm = LLM(
        model=model_name, 
        trust_remote_code=True,
        tensor_parallel_size=1, # Change to the number of GPUs available if you have a multi-GPU setup
        max_model_len = 16384,
        enforce_eager=True,
        gpu_memory_utilization=0.80
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
    answer, docs = answer_with_rag("Permanent pasture producing sheep, lamb (weaned) in united kingdom. cycle description: blue-2019. site description: blue farming system", 
                "What is the functional unit?",
                "The functional unit can either be: \"1 ha\" (one hectare) or \"relative\" (meaning that the quantities "
                "of Inputs and Emissions correspond to the quantities of Products). If the primary product is a crop or "
                "forage, the functional unit must be 1 ha. If \"relative\" is reported above, please also provide the "
                "functional unit most relevant to the production system.", reader, tokenizer, vdb)
    print(answer)
