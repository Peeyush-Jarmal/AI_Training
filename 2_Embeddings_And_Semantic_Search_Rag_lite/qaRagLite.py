from chunker import chunk_text, read_file
from embedder import generate_embeddings
from sentence_transformers import SentenceTransformer
from similarityCheck import similarity_search
from openai import OpenAI
import os



# testing chunker
# sample_text = (
#         "Birds are warm-blooded animals. "
#         "They have feathers and wings. "
#         "Some birds migrate long distances. "
#         "Migration helps birds survive seasonal changes."
#     )
#chunks = chunk_text(sample_text,3,1);
#print(chunks);
userInput = input('Ask me about birds:')
text = read_file("birds_5000_words.txt");
chunks = chunk_text(text,200,15);
#print(len(text));
#print(len(chunks));
embedding_model = SentenceTransformer('all-MiniLM-L6-v2');
dataEmbeddings = generate_embeddings(chunks,embedding_model);
queryEmbeddings =  embedding_model.encode(userInput);
#produce similar vectors for these two sentences
#produce distant vectors for unrelated sentences
#here is where the magic happens the sentenceTransformer responds with similar vectors for similar words.
print(queryEmbeddings);
results = similarity_search(queryEmbeddings,dataEmbeddings,chunks)
for chunk, score in results:
    print(f"\nScore: {score:.3f}")
    print(chunk)

SIMILARITY_THRESHOLD = 0.35
best_score = results[0][1]
print(best_score);
if best_score < SIMILARITY_THRESHOLD:
    print("No details found in the document.")

top_chunks = [chunk for chunk, score in results]
context = "\n\n".join(top_chunks)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = f"""
You must answer the question using ONLY the context below.
If the answer is not present, say "No information available."

Context:
{context}

Question:
{userInput}
"""

def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )
    return response.output_text

answer = ask_llm(prompt);
print(answer);