# Import the Gemini chat model from LangChain, load the API key from the .env file, initialize the Gemini model with a low
# temperature for focused responses, take a prompt from the user, send it to Gemini using LangChain's invoke() method, and
# print the generated response.

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