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
    mode = input(
        "Enter mode of AI: \n"
        "Type A for Angry, F for Funny, or S for Sad: "
    )

    if mode.lower() == "a":
        ai_mode = "You are an angry AI assistant. Give aggressive responses."
    elif mode.lower() == "f":
        ai_mode = "You are a funny AI assistant. Give funny responses."
    elif mode.lower() == "s":
        ai_mode = "You are a sad AI assistant. Give sad responses."
    else:
        ai_mode = "You are a helpful AI assistant. Give normal responses."
    messages = [
        SystemMessage(content=ai_mode)
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