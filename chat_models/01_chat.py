from langchain_google_genai import ChatGoogleGenerativeAI

# To load APIs from environment variable
from dotenv import load_dotenv
load_dotenv()


def getllm():
    return ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2)

def main():
    llm = getllm()
    user_prompt = input("What's in your mind\n")
    response = llm.invoke(user_prompt)
    print(response.content[0]["text"])

main()