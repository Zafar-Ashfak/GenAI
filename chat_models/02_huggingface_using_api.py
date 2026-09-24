# Import Hugging Face's endpoint and chat model classes through LangChain, load the Hugging Face API credentials from the .env
# file, configure the GPT-OSS-20B model using HuggingFaceEndpoint, wrap it with ChatHuggingFace to use it as a chat model.

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0
)

model = ChatHuggingFace(llm = llm)

response = model.invoke("Why we are facing RAM shortage now a days?")

print(response.content)

