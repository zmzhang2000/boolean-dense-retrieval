import os
from tqdm import tqdm
import argparse
import json
import numpy as np
import faiss
import torch
from dataset import get_boolquestions_evalset
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the embedding files of queries and documents.")
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save the retrieval results.")
    parser.add_argument(
        "--encode_queries_online", action="store_true",
        help="Whether to encode queries online.")
    parser.add_argument(
        "--queries_path", type=str, default=None,
        help="Path to the queries file.")
    parser.add_argument(
        '--query_encoder', type=str, required=True,
        help="Name or path of the query encoder model.")
    parser.add_argument(
        "--index_batch_size", type=int, default=10000,
        help="Batch size for building the index.")
    parser.add_argument(
        "--rebuild_index", action="store_true",
        help="Whether to rebuild the index.")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for retrieval.")
    parser.add_argument(
        "--topk", type=int, default=100,
        help="Number of documents to retrieve per query.")
    args = parser.parse_args()
    return args


def embedding_files_sanity_check(file_names):
    file_names_split = [x.split(".") for x in file_names]
    assert len(file_names_split) > 0, f"Invalid file names: {file_names}"
    assert all(len(x) == 4 for x in file_names_split), f"Invalid file names: {file_names}"
    assert len(set(x[0] for x in file_names_split)) == 1, f"Invalid file names: {file_names}"
    assert len(set(x[2] for x in file_names_split)) == 1, f"Invalid file names: {file_names}"
    assert all(str(i) in [x[1] for x in file_names_split] for i in range(int(file_names_split[0][2]))), f"Invalid file names: {file_names}"
    assert all(x[3] == "npy" for x in file_names_split), f"Invalid file names: {file_names}"


def load_query_embeddings(input_path):
    input_path = os.path.join(input_path, "embeddings")
    file_names = os.listdir(input_path)

    qids_files = sorted([x for x in file_names if x.startswith("qids.")])
    embedding_files_sanity_check(qids_files)
    qids = np.concatenate([np.load(os.path.join(input_path, x)) for x in qids_files], axis=0)

    q_embeds_files = sorted([x for x in file_names if x.startswith("q_embeds.")])
    embedding_files_sanity_check(q_embeds_files)
    q_embeds = np.concatenate([np.load(os.path.join(input_path, x))
                               for x in tqdm(q_embeds_files, desc="Loading query embeddings")], axis=0)
    assert len(qids) == len(q_embeds), f"Number of query ids and embeddings do not match: {len(qids)} != {len(q_embeds)}"

    return qids.tolist(), q_embeds


def load_document_embeddings(input_path):
    input_path = os.path.join(input_path, "embeddings")
    file_names = os.listdir(input_path)

    docids_files = sorted([x for x in file_names if x.startswith("docids.")])
    embedding_files_sanity_check(docids_files)
    docids = np.concatenate([np.load(os.path.join(input_path, x)) for x in docids_files], axis=0)

    doc_embeds_files = sorted([x for x in file_names if x.startswith("doc_embeds.")])
    embedding_files_sanity_check(doc_embeds_files)
    doc_embeds = np.concatenate([np.load(os.path.join(input_path, x))
                                 for x in tqdm(doc_embeds_files, desc="Loading document embeddings")], axis=0)
    assert len(docids) == len(doc_embeds), f"Number of document ids and embeddings do not match: {len(docids)} != {len(doc_embeds)}"

    return docids.tolist(), doc_embeds


def index_documents(input_path, index_batch_size=10000, rebuild_index=False):
    index_path = os.path.join(input_path, "index")
    os.makedirs(index_path, exist_ok=True)
    index_file = os.path.join(index_path, "index")
    index_meta_file = os.path.join(index_path, "index_meta.json")
    if os.path.exists(index_file) and os.path.exists(index_meta_file) and not rebuild_index:
        print(f"Loading index from {index_path}.")
        index = faiss.read_index(index_file)
        with open(index_meta_file) as f:
            docids = json.load(f)["docids"]
        print(f"Index loaded. ntotal:", index.ntotal)
    else:
        print("Building index from scratch.")
        docids, doc_embeds = load_document_embeddings(input_path)
        index = faiss.IndexFlatIP(doc_embeds.shape[1])
        doc_embeds_batches = np.array_split(doc_embeds, max(1, len(doc_embeds) // index_batch_size), axis=0)
        for i, doc_embeds_batch in enumerate(tqdm(doc_embeds_batches, desc="Building index")):
            index.add(doc_embeds_batch)
        faiss.write_index(index, index_file)
        with open(index_meta_file, "w") as f:
            json.dump({"docids": docids}, f)
        print("Index built. ntotal:", index.ntotal)

    return index, docids


if __name__ == '__main__':
    args = parse_args()
    if args.encode_queries_online:
        if not (args.queries_path and args.query_encoder):
            raise ValueError("Queries path and query encoder must be provided for online query encoding.")
        if "Qwen" in args.query_encoder:
            model_kwargs = {"torch_dtype":torch.float16}
        else:
            model_kwargs = None
        query_encoder = SentenceTransformer(args.query_encoder, device="cuda", model_kwargs=model_kwargs, trust_remote_code=True)
        eval_qid2query, eval_qid2docids = get_boolquestions_evalset(subset_name=args.queries_path)
        qids = sorted(list(eval_qid2docids.keys()))
        prompt_name = "query" if "Qwen" in args.query_encoder else None
        q_embeds = [query_encoder.encode([eval_qid2query[queryid]], prompt_name=prompt_name) for queryid in tqdm(qids, desc="Encoding queries")]
        q_embeds = np.concatenate(q_embeds, axis=0)
    else:
        qids, q_embeds = load_query_embeddings(args.input_path)
    doc_index, docids = index_documents(args.input_path, args.index_batch_size, args.rebuild_index)

    if args.output_path is None:
        args.output_path = os.path.join(args.input_path, "retrieval")
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, f"results.top{args.topk}.txt"), "w") as f:
        qids_batches = np.array_split(np.array(qids), max(1, len(qids) // args.batch_size), axis=0)
        q_embeds_batches = np.array_split(q_embeds, max(1, len(q_embeds) // args.batch_size), axis=0)
        for qid_batch, q_embeds_batch in tqdm(zip(qids_batches, q_embeds_batches, strict=True),
                                              desc="Retrieving", total=len(qids_batches)):
            D, I = doc_index.search(q_embeds_batch, args.topk)
            returned_docids_list = [[docids[i] for i in doc_indices] for doc_indices in I]

            for qid, returned_docids, returned_scores in zip(qid_batch, returned_docids_list, D, strict=True):
                for rank, (docid, score) in enumerate(zip(returned_docids, returned_scores), start=1):
                    f.write("%s\t%s\t%d\t%f\n" % (qid, docid, rank, score))
