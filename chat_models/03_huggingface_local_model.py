# Load the TinyLlama model locally from Hugging Face using LangChain's HuggingFacePipeline, wrap it as a chat model with
# ChatHuggingFace, send a prompt to the locally running model using invoke(), and print the generated response.

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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