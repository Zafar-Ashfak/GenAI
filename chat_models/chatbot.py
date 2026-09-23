from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()

def getllm():
   llm = HuggingFaceEndpoint(
       repo_id="meta-llama/Llama-3.1-8B-Instruct",
       task="text-generation",
       temperature=0.2
   )

   return ChatHuggingFace(llm=llm)


def main():

    messages = []
    print("-----------------Welcome -----------------  ")
    print("\n\nType exit or quit to close the chat!")

    print("What's in your mind")
    chat_model = getllm()

    while True:
        prompt = input("You: ")
        messages.append(prompt)
        if prompt == "exit" or prompt == "quit":
            break

        response = chat_model.invoke(messages)
        messages.append(response)
        print(f"Bot: {response.content}")

main()