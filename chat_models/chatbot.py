from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from dotenv import load_dotenv
load_dotenv()


def getllm():
   return HuggingFacePipeline(repo_id="openai/gpt-oss-20b")

def get_model():
    llm = getllm()
    return ChatHuggingFace(
        llm = llm
    )

def main():
    print("-----------------Welcome -----------------  ")
    print("\n\nType exit or quit to close the chat!")

    print("What's in you mind")

    while True:
        prompt = input("You: ")
        if prompt == "exit" or prompt == "quit":
            break

        chat_model = get_model()
        response = chat_model.invoke(prompt)
        print(f"Bot: {(response.content[0]["text"])}")

main()