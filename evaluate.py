import rag_retrieval
import constants
from langchain_community.vectorstores import FAISS
import json
import evaluate


def eval_models(dataset, dataset_name, RAG=True):
    # for each model, we need to run an evaluation
    model_names = []
    for i in model_names:
        # configure model
        reader, tokenizer = rag_retrieval.model_config()
        print("model configured")
        labels = []
        preds = []

        # iterate through all questions in the dataset
        for question in dataset:
            # extract question, label, and category
            label = question["referenceResponse"]
            category = question["category"]
            prompt = question["prompt"]

            if RAG:
                print(str(i) + " with RAG on dataset " + dataset_name)
                answer, docs = rag_retrieval.answer_with_rag(prompt, reader, tokenizer, vdb)
            else:
                print(str(i) + " without RAG on dataset " + dataset_name)
                answer = rag_retrieval.answer_without_rag(prompt, reader, tokenizer)

            preds.append(answer)
            labels.append(label)

        # compare answer to expected list of answers
        metric = evaluate.load("accuracy")
        metric.compute(predictions=preds, references=labels)



if __name__ == "__main__":
    # load vdb information
    embeddings = constants.EMBED_MODEL
    vdb = FAISS.load_local(
        constants.VDB_LOCATION, embeddings, allow_dangerous_deserialization=True)
    print("vdb loaded")

    # load qa dataset
    filenames = ["data/recalculated/qa_dataset.jsonl", "data/qa_dataset.jsonl"]
    for k in filenames:
        with open(k, 'r', encoding='utf-8') as f:
            data = json.load(f)
            eval_models(data, k, True)
            eval_models(data, k, False)

