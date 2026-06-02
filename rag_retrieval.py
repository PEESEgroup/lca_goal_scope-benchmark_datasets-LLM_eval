import os
import sys

# Force the OS to prioritize Conda's modern C++ libraries
os.environ["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# might need to use the following to run on command line in AWS: 
# NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=lo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH /opt/conda/bin/python /home/sagemaker-user/llm-goal-scope/rag_retrieval.py

from vllm import LLM, SamplingParams
from transformers import Pipeline, pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import constants
from tqdm import tqdm
from typing import Any, List, Tuple


def answer_with_rag(
        system_description: list[str],
        question: str,
        hestia: str,
        llm: Any,  # Your vLLM LLM engine instance
        reading_tokenizer: Any,
        knowledge_index: Any,
        rerank_model: CrossEncoder,
        num_retrieved_docs: int = 15,
        num_docs_final: int = 3,
        num_tokens: int = 256,
        temperature: float = 0.0
) -> tuple[list[str], list[list[str]]]:
    """
    Batched method to answer a list of queries simultaneously using vLLM continuous batching.
    """
    print(f"Sending {len(system_description)} queries to FAISS index...")
    
    # Check if the LangChain FAISS index object natively supports batching
    if hasattr(knowledge_index, "batch_search"):
        # Define a reasonable batch chunk size (e.g., 32 or 64 queries at a time)
        FAISS_BATCH_SIZE = 64  
        all_retrieved_docs = []

        # Loop through your descriptions in chunks with a visual progress bar
        for i in tqdm(range(0, len(system_description), FAISS_BATCH_SIZE), desc="FAISS Batch Searching"):
            mini_batch = system_description[i : i + FAISS_BATCH_SIZE]
            
            # Query the vector index with the current chunk
            batch_results = knowledge_index.batch_search(mini_batch, k=num_retrieved_docs)
            
            # Extend our master collection list
            all_retrieved_docs.extend(batch_results)
    else:
        # Fallback: Loop manually but isolate thread safety issues
        import os
        # Prevent FAISS from aggressively over-allocating internal CPU threads per query
        os.environ["OMP_NUM_THREADS"] = "1" 
        os.environ["MKL_NUM_THREADS"] = "1"
        
        all_retrieved_docs = [
            knowledge_index.similarity_search(desc, k=num_retrieved_docs) 
            for desc in system_description
        ]
    print("Gathered All Documents")

    final_prompts = []
    all_reranked_docs_text = []
    
    # Configure generation parameters up front
    sampling_params = SamplingParams(max_tokens=num_tokens, temperature=temperature)

    # Process retrieval & reranking per description item
    for desc, docs in tqdm(zip(system_description, all_retrieved_docs)):
        doc_texts = [doc.page_content for doc in docs]
        
        # Build text-matching pairs for this specific description query
        pairs = [[f"{desc} {question}", text] for text in doc_texts]
        scores = rerank_model.predict(pairs)

        # Sort documents based on CrossEncoder scores
        scored_docs = sorted(zip(scores, doc_texts), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:num_docs_final]]
        all_reranked_docs_text.append(top_docs)

        # Build the context block for this single query profile
        context_str = "\nAdditional Context:\n" + "".join(
            [f"Source {i}:::\n{doc}\n" for i, doc in enumerate(top_docs)]
        )

        # format using the chat template sequence
        chat_structure = [
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
            Context: {context_str}"""
        },
    ]

        # Convert to raw structural prompt token strings
        raw_prompt = reading_tokenizer.apply_chat_template(
            chat_structure, tokenize=False, add_generation_prompt=True
        )
        final_prompts.append(raw_prompt)

    print("Reranked All Documents")

    MINI_BATCH_SIZE = 128  # Large enough to max out the GPUs, small enough for regular updates
    outputs = []

    # Loop through the prompts in steps of MINI_BATCH_SIZE
    for i in tqdm(range(0, len(final_prompts), MINI_BATCH_SIZE), desc="vLLM Generating Answers"):
        mini_batch = final_prompts[i : i + MINI_BATCH_SIZE]
        
        # Generate for the current chunk
        batch_outputs = llm.generate(mini_batch, sampling_params)
        outputs.extend(batch_outputs)
    
    # Collect answers corresponding cleanly to the batch items
    generated_answers = [out.outputs[0].text.strip() for out in outputs]
    
    return generated_answers, all_reranked_docs_text


def model_config(model_name="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"):
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
    
    # trained on a ml.g6.48xlarge with 8x24GB VRAM GPUs
    llm = LLM(
        model=model_name, 
        trust_remote_code=True,
        load_format="safetensors",
        tensor_parallel_size=8,
        max_model_len=16384,
        gpu_memory_utilization=0.6,
        max_num_seqs=1,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        kv_cache_dtype="fp8",
        quantization="compressed-tensors"
    )

    return llm, tokenizer


if __name__ == "__main__":
    # NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=lo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH /opt/conda/bin/python /home/sagemaker-user/llm-goal-scope/rag_retrieval.py
    embeddings = constants.EMBED_MODEL
    os.chdir('llm-goal-scope')
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")
    reader, tokenizer = model_config()
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("model configured")
    # test question to make sure things are working
    question = "what is a functional unit for sheep production in the UK?"
    answer, docs = answer_with_rag(["Permanent pasture producing sheep, lamb (weaned) in united kingdom. cycle description: blue-2019. site description: blue farming system"], 
                "What is the functional unit?",
                "The functional unit can either be: \"1 ha\" (one hectare) or \"relative\" (meaning that the quantities "
                "of Inputs and Emissions correspond to the quantities of Products). If the primary product is a crop or "
                "forage, the functional unit must be 1 ha. If \"relative\" is reported above, please also provide the "
                "functional unit most relevant to the production system.", reader, tokenizer, vdb, rerank_model)
    print(answer)
