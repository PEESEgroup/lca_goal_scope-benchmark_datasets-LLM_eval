import constants
from langchain_community.vectorstores import FAISS
import json
from evaluate import evaluator
from datasets import Dataset, load_dataset


def eval_models(dataset, dataset_name):
    # for each model, we need to run an evaluation
    model_names = ["distilbert-base-uncased-distilled-squad"]
    for i in model_names:
        # as qa
        # TODO: filter by type???
        task_evaluator = evaluator("question-answering")
        print("evaluating dataset", dataset_name, " using model", i)
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad", #TODO: is the metric having difficulty due to the use of "squad". what others are available?
            strategy="bootstrap",
            n_resamples=30
        )


        """
        # TODO: try text-generation
        task_evaluator = evaluator("text-generation")
        print("evaluating dataset", dataset_name, " using model", i)
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad",
            strategy="bootstrap",
            input_column="question",
            label_column="answers",
            n_resamples=30
        )
        """

        """
        # for text2text-generation, which may or may not be what we want
        task_evaluator = evaluator("text2text-generation")
        print("evaluating dataset", dataset_name, " using model", i)
        eval_results = task_evaluator.compute(
            model_or_pipeline=i,
            data=dataset,
            metric="squad",
            strategy="bootstrap",
            n_resamples=30
        )

        print(dataset_name, eval_results)
        """

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
        #TODO: rerun rag code
        dataset = Dataset.from_list(data)
        data = load_dataset("squad", split="validation[:2]")
        print(data[0])
        print(dataset[0])
        print("dataset loaded")
        eval_models(dataset, k)

