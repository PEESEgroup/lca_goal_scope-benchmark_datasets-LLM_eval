from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"}, #TODO: find and set appropriate device when running later (non-locally)
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    show_progress=True
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
VDB_LOCATION = "./vectorstore/vs_goalscope"