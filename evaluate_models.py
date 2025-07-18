import constants
from langchain_community.vectorstores import FAISS
import json
from evaluate import evaluator
from datasets import Dataset, load_dataset


def eval_models(dataset, dataset_name):
    # for each model, we need to run an evaluation
    model_names = ["distilbert-base-uncased-distilled-squad"]
    for i in model_names:
        task_evaluator = evaluator("question-answering")
        print("evaluating dataset", dataset_name, " using model", i)
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
    filenames = ["data/recalculated/qa_dataset.jsonl", "data/qa_dataset.jsonl", "data/recalculated/rag_qa_dataset.jsonl", "data/rag_qa_dataset.jsonl"]
    for k in filenames:
        data = []
        with open(k, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        # convert to dataset
        dataset = Dataset.from_list(data)
        print("dataset loaded")
        eval_models(dataset, k)

