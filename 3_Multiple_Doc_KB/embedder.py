from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts)


def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=3):
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    top_results = scores.topk(k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "score": float(score),
            "text": chunks[idx]["text"],
            "doc_id": chunks[idx]["doc_id"],
            "chunk_id": chunks[idx]["chunk_id"]
        })
    return results