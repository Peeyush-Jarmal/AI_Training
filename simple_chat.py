
# ✅ LLM APIs
# ✅ System vs user prompts
# ✅ Hallucination control
# ✅ Context grounding
# ✅ Token usage basics
# ✅ Python integration

import os;
#from dotenv import load_dotenv;
from openai import OpenAI ;

#load_dotenv()
print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))



user_input = input("Enter Your Prompt: ");
with open("birds.txt", "r", encoding="utf-8") as file:
    content = file.read()

#print(content);
##

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"));
response = client.chat.completions.create(
  model="gpt-5-nano",
  messages=[
      {
          "role": "system",
          "content": (
              f"You are helpful assistant that answers questions about birds based on the context provided. \n\nContext: {content} "
              "If the answer is not explicitly present in the context, say: "
              "'I cannot answer this question based on the provided document.' "
              "Do not use outside knowledge or make assumptions."
          )
      },
      {
            "role": "user",
            "content": user_input
      }

  ]
)
print(response.choices[0].message.content);