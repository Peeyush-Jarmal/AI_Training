import os
from chunker import chunk_text
from embedder import embed_texts,retrieve_top_k
from openai import OpenAI


def load_documents(folder_path):
    documents = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file)) as f:
                documents[file] = f.read()
    return documents


all_chunks = []
def createChunksFromDocs(folder_path):
    documents = load_documents(folder_path) 
    for doc_id, text in documents.items():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk
            })

createChunksFromDocs('./');  
#print(all_chunks);          

texts = [c["text"] for c in all_chunks]
embeddings = embed_texts(texts)
#print(embeddings);
####The golden rule####
#Embeddings never carry metadata.
#Metadata is carried by YOU.
#Alignment is the contract.

#embed user query
query = "Do birds fly?"
def embed_query(query):
    return embed_texts(query)

queryEmbedding = embed_query(query);

#all_chunks has to passed retrieve_top_k , this will rank best match and send the chunk data.
# Since the  queryEmbedding[1] will be mapped to all_chunks[1]
#print(all_chunks);
top_chunks = retrieve_top_k(queryEmbedding,embeddings,all_chunks,k=3)
#print(top_chunks);

MIN_SCORE = 0.3

if top_chunks[0]["score"] < MIN_SCORE:
    print("No relevant information found.")
    exit()

def build_context(top_chunks):
    context = ""
    for item in top_chunks:
        context += (
            f"[Source: {item['doc_id']} | Chunk {item['chunk_id']}]\n"
            f"{item['text']}\n\n"
        )
    return context    

context = build_context(top_chunks)
#print(context)

prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )
    return response.output_text

answer = ask_llm(prompt);
print(answer);