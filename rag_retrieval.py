from transformers import Pipeline, pipeline
from typing import Any
import torch
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import constants
import os


def answer_with_rag(
        question: str,
        llm: Pipeline,
        reading_tokenizer: AutoTokenizer,
        knowledge_index: FAISS,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
) -> tuple[Any, list[str]]:
    """
    method to answer a given query using RAG data
    :param question: input string query
    :param llm: LLM pipeline
    :param reading_tokenizer: tokenizer for dataset
    :param knowledge_index: vector database
    :param num_retrieved_docs: number of retrieved documents
    :param num_docs_final: number of finalized retrieved documents sent to LLM
    :return: the generated answer and the relevant documents
    """

    # configure rag prompt
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """You are an expert on agricultural life cycle assessment (LCA). 
            Please summarize the life cycle assessment information that is relevant to the context.
            Please use as few words as necessary.
            You do not need to provide document numbers or restate parts of the prompt.""",
        },
        {
            "role": "user",
            "content": """Context:
            {context}
            ---
            Question: {question}""",
        },
    ]

    RAG_PROMPT_TEMPLATE = reading_tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    # Gather documents with retriever
    print(f"=> Retrieving documents for query {question}...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text
    # relevant_docs = relevant_docs[:num_docs_final]

    # reranking documents
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # pair the query with each document and calculate scores
    pairs = [[question, doc] for doc in relevant_docs]
    scores = rerank_model.predict(pairs)

    # rank scores
    scored_docs = sorted(zip(scores, relevant_docs), key=lambda x: x[0], reverse=True)

    # use a counter to save the top n docs
    counter = 0
    reranked_docs = []

    # go through the documents and print out the top ones
    for score, doc in scored_docs:
        # print(f"Score: {score:.4f} | Doc: {doc}")
        counter = counter + 1

        # append the top documents to a list
        if counter <= num_docs_final:
            reranked_docs.append(doc)

    # build the final prompt
    context = "\nAdditional Context:\n"
    context += "".join([f"Source {str(i)}:::\n" + doc for i, doc in enumerate(reranked_docs)])
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # retrieve an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    # for llama: llm(final_prompt)[0]["generated_text"][-1]['content']???

    # do some string processing to extract just the generated string
    generated_answer = answer.split("<|start_header_id|>assistant<|end_header_id|>")[1]
    generated_answer = generated_answer.strip()
    print(f"=> model answers \"{generated_answer}\"\n\n")
    
    return generated_answer, relevant_docs


def model_config(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """
    set up LLM model config for summarizing RAG results
    :param model_name: name of the model - in our case Llama3.2-3B-Instruct
    :return: the LLM pipeline and the tokenizer
    """
    # "HuggingFaceH4/zephyr-7b-beta" for testing
    # "meta-llama/Llama-3.2-3B-Instruct" for actuality
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        dtype=torch.bfloat16,
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
    answer, docs = answer_with_rag(question, reader, tokenizer, vdb)
    print(answer)
