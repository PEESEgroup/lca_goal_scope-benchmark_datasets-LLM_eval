import datasets
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
import constants
import os
import json


def split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: str,
        markdown_separators: List[str]
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=markdown_separators,
    )

    docs_processed = []
    print("\nchunking documents")
    for doc in tqdm(knowledge_base):
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    print("\nremoving duplicates")
    for doc in tqdm(docs_processed):
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def get_rag_json():
    file_path = "./data/RAG-textract/"
    rag_data = []

    counter = 0

    # fix cwd problems
    if os.getcwd() == "/home/sagemaker-user":
        os.chdir("./llm-goal-scope")

    for entry_name in tqdm(os.listdir(str(file_path))):
        counter = counter + 1
        # get file path
        fname = os.path.join(file_path, entry_name)
        # open file
        with open(fname, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                pretty_json_string = json.dumps(data).replace("- ", "")  # remove hyphens over lines
                ds_data = {"source": entry_name,
                           "text": pretty_json_string}
                rag_data.append(ds_data)
            except UnicodeDecodeError:
                print("cannot read the file, skipping")
                continue  # can't read the file, so skipping
    
    rag_data = {"data": rag_data}
    with open('./rag_data.json', 'w') as f:
        json.dump(rag_data, f)


def vs_creation(filename, embedding_model, EMBEDDING_MODEL_NAME):
    get_rag_json()
    ds = load_dataset("json", data_files="./rag_data.json", field="data")
    print("\ndataset loaded")

    # dataset as a list of dicts
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
    ]

    # chunking documents
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
        "<*>",
        "[",
        "{",
    ]

    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
        markdown_separators=MARKDOWN_SEPARATORS
    )
    print("\nplotting documents")

    # Let's visualize the chunk sizes we would have in tokens from a common model
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()

    # embed documents
    vdb = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    print("\ndocuments embedded")

    # save embeddings locally
    vdb.save_local(filename)
    print("vector store saved locally")


if __name__ == "__main__":
    embed_model = constants.EMBED_MODEL
    embed_model_name = constants.EMBEDDING_MODEL_NAME
    vs_creation(constants.VDB_LOCATION, embed_model, embed_model_name)
