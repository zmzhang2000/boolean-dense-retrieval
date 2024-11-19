from datasets import load_dataset


HF_REPO_NAME = "ustc-zhangzm/BoolQuestions"


def get_boolquestions_docid2doc(subset_name):
    print("Loading corpus for", subset_name)
    corpus = load_dataset(HF_REPO_NAME, name=f"{subset_name}-corpus", split="corpus")
    if subset_name == "MSMARCO":
        docid2doc = {sample["docid"]: sample["doc"] for sample in corpus}
    elif subset_name == "NaturalQuestions":
        docid2doc = {
            sample["docid"]: f'{sample["title"]} [SEP] {sample["doc"]}'
            for sample in corpus
        }
    else:
        raise ValueError(f"Unsupported BoolQuestions subset name: {subset_name}")
    return docid2doc


def get_boolquestions_evalset(
    subset_name, subset_question_types="all", return_negatives=False
):
    print("Loading evaluation set for", subset_name)
    if isinstance(subset_question_types, str):
        subset_question_types = [subset_question_types]
    if subset_question_types == ["all"]:
        subset_question_types = ["and", "or", "not"]
    
    eval_data = load_dataset(HF_REPO_NAME, name=subset_name, split="eval")
    qid2query = {}
    qid2docids = {}
    for d in eval_data:
        if d["question_type"] not in subset_question_types:
            continue
        qid = d["qid"]
        qid2query[qid] = d["question"]
        if return_negatives:
            qid2docids[qid] = [neg_ctx["passage_id"] for neg_ctx in d["negative_ctxs"]]
        else:
            qid2docids[qid] = [pos_ctx["passage_id"] for pos_ctx in d["positive_ctxs"]]
    return qid2query, qid2docids
