import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
from dataset import get_boolquestions_docid2doc, get_boolquestions_evalset
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--query_encoder', type=str, required=True,
    help="Name or path of the query encoder model.")
parser.add_argument(
    '--document_encoder', type=str, required=True,
    help="Name or path of the document encoder model.")
parser.add_argument(
    '--data', type=str, required=True,
    help="Name or path of the data to be inferred.")
parser.add_argument(
    '--output_path', type=str, default="outputs",
    help="Path of the output file.")
parser.add_argument(
    '--device',
    type=str, default="cuda",
    help="Device to run the model on.")
parser.add_argument(
    "--num_shards",
    type=int, default=1,
    help="Number of shards to split the dataset into.")
parser.add_argument(
    "--shard_id",
    type=int, default=0,
    help="ID of the shard to run the inference on.")
parser.add_argument(
    "--encode_query",
    action='store_true',
    help="Whether to encode the query or the document.")
args = parser.parse_args()


if __name__ == '__main__':
    assert args.num_shards > 0 and args.shard_id < args.num_shards, "Invalid shard configuration"
    os.makedirs(args.output_path, exist_ok=True)

    if "Qwen" in args.document_encoder:
        model_kwargs = {"torch_dtype":torch.float16}
    else:
        model_kwargs = None
    document_encoder = SentenceTransformer(args.document_encoder, device=args.device, model_kwargs=model_kwargs, trust_remote_code=True)
    batch_size = 8 if "Qwen" in args.document_encoder else 128

    if args.encode_query:
        eval_qid2query, eval_qid2docids = get_boolquestions_evalset(subset_name=args.data)

        qids_file = os.path.join(args.output_path, f"qids.{args.shard_id}.{args.num_shards}.npy")
        q_embeds_file = os.path.join(args.output_path, f"q_embeds.{args.shard_id}.{args.num_shards}.npy")

        if os.path.exists(qids_file) and os.path.exists(q_embeds_file):
            print(f"Embeddings for queries already exist in {q_embeds_file}. Skip encoding.")
        else:
            if args.query_encoder == args.document_encoder:
                query_encoder = document_encoder
            else:
                query_encoder = SentenceTransformer(args.query_encoder, device=args.device, model_kwargs=model_kwargs, trust_remote_code=True)

            qids = sorted(list(eval_qid2docids.keys()))
            shard_indices = np.array_split(np.arange(len(qids)), args.num_shards)[args.shard_id]
            qids = np.array(qids)[shard_indices].tolist()

            embeds = []
            batched_query = []
            for idx, queryid in enumerate(tqdm(qids, desc="Encoding queries")):
                batched_query.append(eval_qid2query[queryid])
                if len(batched_query) == batch_size or idx == len(qids) - 1:
                    embed = query_encoder.encode(batched_query)
                    embeds.append(embed)
                    batched_query = []
            embeds = np.concatenate(embeds, axis=0)

            np.save(qids_file, np.array(qids))
            np.save(q_embeds_file, embeds)

    docid2doc = get_boolquestions_docid2doc(subset_name=args.data)
    docids_file = os.path.join(args.output_path, f"docids.{args.shard_id}.{args.num_shards}.npy")
    doc_embeds_file = os.path.join(args.output_path, f"doc_embeds.{args.shard_id}.{args.num_shards}.npy")

    if os.path.exists(docids_file) and os.path.exists(doc_embeds_file):
        print(f"Embeddings for documents already exist in {doc_embeds_file}. Skip encoding.")
    else:
        docids = sorted(list(docid2doc.keys()))
        shard_indices = np.array_split(np.arange(len(docids)), args.num_shards)[args.shard_id]
        docids = np.array(docids)[shard_indices].tolist()

        embeds = []
        batched_doc = []
        for idx, docid in enumerate(tqdm(docids, desc="Encoding documents")):
            batched_doc.append(docid2doc[docid])
            if len(batched_doc) == batch_size or idx == len(docids) - 1:
                embed = document_encoder.encode(batched_doc)
                embeds.append(embed)
                batched_doc = []
        embeds = np.concatenate(embeds, axis=0)

        np.save(docids_file, np.array(docids))
        np.save(doc_embeds_file, embeds)
