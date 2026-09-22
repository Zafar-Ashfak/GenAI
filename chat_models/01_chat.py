from langchain_google_genai import ChatGoogleGenerativeAI

# To load APIs from environment variable
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.8)

response = model.invoke("Tell me everything about Taj Mahal?")

print(response.content)