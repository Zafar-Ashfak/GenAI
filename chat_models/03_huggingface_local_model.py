from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

from dotenv import load_dotenv
load_dotenv()

print("HF_TOKEN loaded:", os.getenv("HF_TOKEN") is not None)

print("Loading model...")

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

print("Model loaded. Generating response...")

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("Tell me everything about apple company.")

print("Response received:")
print(response.content)