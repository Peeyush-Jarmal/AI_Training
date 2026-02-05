import numpy as np

def generate_embeddings(chunks, embedding_model):
    embeddings = []
    for chunk in chunks:
        emb = embedding_model.encode(chunk)  # placeholder
        embeddings.append(emb)
    return np.array(embeddings)