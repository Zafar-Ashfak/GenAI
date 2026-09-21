# To load APIs from environment variable

# from langchain.chat_models import init_chat_model
# from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

response = model.invoke("What is Agentic AI?")

print(response.content)