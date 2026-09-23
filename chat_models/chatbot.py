from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

    messages = [

        SystemMessage(content="""
            You are a helpful AI assistant.
            Your job is to answer the user's questions clearly, accurately, and in simple language.
            Give direct answers and avoid unnecessary information.
            For technical questions, explain concepts with simple examples when useful.
            If you are unsure about something, clearly say that you are unsure instead of making up information.
            Maintain a friendly and professional tone.
            """
        )
    ]
    print("-----------------Welcome -----------------  ")
    print("\n\nType exit or quit to close the chat!")

    print("What's in your mind")
    chat_model = getllm()

    while True:
        prompt = input("You: ")
        messages.append(HumanMessage(content=prompt))
        if prompt == "exit" or prompt == "quit":
            break

        response = chat_model.invoke(messages)
        messages.append(AIMessage(content=response.content))
        print(f"Bot: {response.content}")

    print(f"__________________________________"
          f"\n\n\n{messages}")

main()