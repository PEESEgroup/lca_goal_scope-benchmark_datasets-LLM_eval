import constants
from langchain_community.vectorstores import FAISS
import json
from evaluate import evaluator


def eval_models(dataset, dataset_name):
    # for each model, we need to run an evaluation
    model_names = ["distilbert-base-uncased-distilled-squad"]
    for i in model_names:
        task_evaluator = evaluator("question-answering")
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad",
            strategy="bootstrap",
            n_resamples=30
        )

        print(dataset_name, eval_results)



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
            eval_models(data, k)
            eval_models(data, k)

