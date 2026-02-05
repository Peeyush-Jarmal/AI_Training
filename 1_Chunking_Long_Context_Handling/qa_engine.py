from chunker import read_file, chunk_text
from retriever import retrieve_chunk
from openai import OpenAI
import os 

text = read_file("birds_5000_words.txt")
print(len(text))
chunks = chunk_text(text)
print(len(chunks))

question = input("Ask a question: ")

context = retrieve_chunk(chunks, question)
if context:
    print(len(context))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )
    return response.output_text

prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\n\nIf the answer is not explicitly present in the context, say: 'I cannot answer this question based on the provided document.' Do not use outside knowledge or make assumptions."

#print(ask_llm(prompt))
