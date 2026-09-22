from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0
)

model = ChatHuggingFace(llm = llm)

response = model.invoke("What is AI, ML and DL?")

print(response.content)

