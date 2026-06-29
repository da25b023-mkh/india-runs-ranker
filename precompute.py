"""
precompute.py — Run ONCE offline. No time limit.
GPU-forced version using torch directly.
"""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from features import _career_text


def load_candidates(path: str):
    candidates = []
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        else:
            candidates = json.load(f)
    return candidates


def build_jd_text() -> str:
    return """
    Senior AI Engineer founding team Redrob AI Series A talent intelligence platform.
    Production experience embeddings based retrieval systems sentence transformers
    semantic search dense retrieval vector databases Pinecone Weaviate Qdrant Milvus
    FAISS Elasticsearch hybrid search BM25 ranking reranking NDCG MRR MAP evaluation
    frameworks learning to rank LTR information retrieval NLP natural language processing
    Python MLOps model serving inference production deployment A/B testing online evaluation
    recommendation systems search ranking 5 to 9 years experience product company startup
    scrappy shipper not pure research not consulting only Pune Noida India LLM fine tuning
    LoRA QLoRA PEFT open source contributions applied ML engineer
    """


def encode_on_gpu(model, texts, batch_size=64):
    """Manually encode on GPU using torch directly."""
    device = torch.device("cuda")
    model = model.to(device)
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            encoded = model.tokenize(batch)
            # Only move tensor values to GPU, skip strings
            encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in encoded.items()}
            output = model.forward(encoded)
            embeddings = output["sentence_embedding"]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="candidates.jsonl")
    parser.add_argument("--output", default="precomputed")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading candidates...")
    candidates = load_candidates(args.input)
    print(f"Loaded {len(candidates):,} candidates")

    print("Loading model: BAAI/bge-large-en-v1.5")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Model loaded ✓")

    print("Encoding job description...")
    jd_text = build_jd_text()
    jd_emb = encode_on_gpu(model, [jd_text], batch_size=1)[0]
    np.save(os.path.join(args.output, "jd_embedding.npy"), jd_emb)
    print("JD embedding saved.")

    print("Building candidate texts...")
    candidate_ids = []
    texts = []
    for c in tqdm(candidates):
        candidate_ids.append(c["candidate_id"])
        texts.append(_career_text(c))

    print(f"Encoding {len(texts):,} candidates on GPU...")
    all_embeddings = encode_on_gpu(model, texts, batch_size=64)

    ids_array = np.array(candidate_ids)
    np.save(os.path.join(args.output, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(args.output, "candidate_ids.npy"), ids_array)

    print(f"\nDone. Saved to {args.output}/")
    print(f"  embeddings.npy    shape: {all_embeddings.shape}")
    print(f"  candidate_ids.npy shape: {ids_array.shape}")
    print(f"  jd_embedding.npy  shape: {jd_emb.shape}")


if __name__ == "__main__":
    main()
