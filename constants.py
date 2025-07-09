from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu"}, #TODO: find and set appropriate device when running later (non-locally)
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    show_progress=True
)
VDB_LOCATION = "/vectorstore/vs_goalscope"