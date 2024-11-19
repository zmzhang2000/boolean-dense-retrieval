import json
import argparse
from dataset import get_boolquestions_evalset
from sentence_transformers.evaluation import InformationRetrievalEvaluator


parser = argparse.ArgumentParser()
parser.add_argument(
    '--results_file', type=str, required=True,
    help="Path to the results file.")
parser.add_argument(
    '--data', type=str, required=True,
    help="Name or path of the raw data.")
args = parser.parse_args()


all_metrics = {}
for question_type in ["all", "and", "or", "not"]:
    for split in ["pos", "neg"]:
        eval_qid2query, eval_qid2docids = get_boolquestions_evalset(
            subset_name=args.data, subset_question_types=question_type, 
            return_negatives=True if split == "neg" else False
        )

        print("Evaluating question types:", question_type)
        print("Number of evaluated questions:", len(eval_qid2docids))

        evaluator = InformationRetrievalEvaluator(queries=eval_qid2query,
                                                corpus={},
                                                relevant_docs=eval_qid2docids,
                                                accuracy_at_k=[1, 3, 5, 10, 20, 100],
                                                mrr_at_k=[10])
        qid2idx = {qid: idx for idx, qid in enumerate(evaluator.queries_ids)}
        queries_result_list = [[] for _ in range(len(qid2idx))]
        with open(args.results_file) as f:
            for line in f:
                qid, docid, rank, score = line.strip().split("\t")
                if qid in qid2idx:
                    queries_result_list[qid2idx[qid]].append({"corpus_id": docid, "score": float(score)})

        metrics = evaluator.compute_metrics(queries_result_list)
        all_metrics[f"{question_type}_{split}"] = metrics

with open(args.results_file.replace(".txt", ".metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)
from pprint import pprint
pprint(all_metrics)
