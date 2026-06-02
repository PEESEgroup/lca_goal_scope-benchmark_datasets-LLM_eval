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
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np

# Create a worker function that encapsulates the call
def worker_search(desc, index_instance, k):
    return index_instance.similarity_search(desc, k=k)


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
    system_description = system_description[:10]
    print(f"Sending {len(system_description)} queries to FAISS index...")

    # Check if the LangChain FAISS index object natively supports batching
    if hasattr(knowledge_index, "batch_search"):
        # Define a reasonable batch chunk size (e.g., 32 or 64 queries at a time)
        FAISS_BATCH_SIZE = 64  
        all_retrieved_docs = []

        # Loop through your descriptions in chunks with a visual progress bar
        for i in tqdm(range(0, len(system_description), FAISS_BATCH_SIZE), desc="FAISS Batch Searching"):
            mini_batch = system_description[i : i + FAISS_BATCH_SIZE]
            print(mini_batch)
            combined_mini_batch = [
                f"Description: {desc.strip()} Question: {question.strip()}" 
                for desc in mini_batch_desc
            ]
            
            # Query the vector index with the current chunk
            batch_results = knowledge_index.batch_search(mini_batch, k=num_retrieved_docs)
            
            # Extend our master collection list
            all_retrieved_docs.extend(batch_results)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        # Uncap the OpenMP threads so individual workers can compute distances fast
        # A sweet spot for parallel processes is 2 to 4 threads per worker process.
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        # Suppress status updates and download bar streams
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        # combine your descriptions and questions into a flat text list
        combined_queries = [
            f"Description: {desc.strip()} Question: {question.strip()}" 
            for desc in system_description
        ]

        print(combined_queries[0])

        # Extract the underlying embedding engine from LangChain
        embedding_engine = knowledge_index.embedding_function

        # Matrix Vectorization (Runs across all cores simultaneously) passing the entire list to embed_documents lets the C++/PyTorch backend 
        # process the data as a single parallel tensor batch.
        print("Vectorizing entire query batch...")
        query_embeddings = embedding_engine.embed_documents(combined_queries)

        # Convert to a contiguous float32 numpy array for the native FAISS C++ layer
        query_embeddings_matrix = np.array(query_embeddings, dtype=np.float32)

        # ative FAISS Index Batch Search This uses the underlying C++ index to search all 851 vectors at the same time,
        print("Querying FAISS C++ Index...")
        scores, indices = knowledge_index.index.search(query_embeddings_matrix, num_retrieved_docs)

        # Reconstruct the LangChain Document objects to keep downstream code intact
        all_retrieved_docs = []
        for row_indices in tqdm(indices):
            doc_batch = []
            for idx in row_indices:
                if idx == -1:
                    continue  # FAISS returns -1 if fewer documents exist than requested k
                
                # Pull the original Document object out of the LangChain store maps
                doc_id = knowledge_index.index_to_docstore_id[idx]
                doc = knowledge_index.docstore.search(doc_id)
                doc_batch.append(doc)
            all_retrieved_docs.append(doc_batch)

    os.environ["HF_HUB_OFFLINE"] = "0"
    # Suppress status updates and download bar streams
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "0"
    print("Gathered All Documents")

    final_prompts = []
    
    # Configure generation parameters up front
    sampling_params = SamplingParams(
        max_tokens=num_tokens, 
        temperature=temperature, 
        stop=[
            "<|end_header_id|>", 
            "<|eot_id|>"
        ])

    # Process retrieval & reranking per description item
    for desc, docs in tqdm(zip(system_description, all_retrieved_docs)):
        doc_texts = [doc.page_content for doc in docs]
        
        # Build text-matching pairs for this specific description query
        pairs = [[f"Description: {desc} Question: {question}", text] for text in doc_texts]
        scores = rerank_model.predict(pairs)

        # Sort documents based on CrossEncoder scores
        scored_docs = sorted(zip(scores, doc_texts), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:num_docs_final]]

        # Build the context block for this single query profile
        context_str = "".join(
            [f"Source {i}:::\n{doc}\n" for i, doc in enumerate(top_docs)]
        )

        # format using the chat template sequence
        chat_structure = [
            {
                "role": "system",
                "content": """
                "You are an expert on agricultural life cycle assessment (LCA). "
                "Your task is to provide the final answer to the user's question, "
                "evaluated strictly against the provided context and HESTIA schema guidelines.\n\n"
                """,
            },
            {
                "role": "user",
                "content": f"""HESTIA schema information: {hestia}
                ---
                Global Task Question: {question}
                ---
                Target Evaluation Description: {desc}
                ---
                Retrieved Context Blocks: {context_str}
                ---
                Respond strictly in the following format:
                Additional Relevant Context: [Insert answer here]"""
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
    
    return generated_answers


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
        max_model_len=32768,             
        gpu_memory_utilization=0.70, 
        enable_prefix_caching=True,      
        enable_chunked_prefill=True,     
        max_num_seqs=16,
        enforce_eager=False,             
        disable_custom_all_reduce=False, 
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
    answer = answer_with_rag(["Permanent pasture producing sheep, lamb (weaned) in united kingdom. cycle description: blue-2019. site description: blue farming system"], 
                "What is the functional unit?",
                "The functional unit can either be: \"1 ha\" (one hectare) or \"relative\" (meaning that the quantities "
                "of Inputs and Emissions correspond to the quantities of Products). If the primary product is a crop or "
                "forage, the functional unit must be 1 ha. If \"relative\" is reported above, please also provide the "
                "functional unit most relevant to the production system.", reader, tokenizer, vdb, rerank_model)
    print(answer)
