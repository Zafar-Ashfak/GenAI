import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

MODES = {
    "Normal": "You are a helpful AI assistant. Give normal responses.",
    "Angry": "You are an angry AI assistant. Give aggressive responses.",
    "Funny": "You are a funny AI assistant. Give funny responses.",
    "Sad": "You are a sad AI assistant. Give sad responses.",
}


@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


st.title("Meta Bot")
st.write("Llama AI chatbot with Normal, Angry, Funny, and Sad modes.")

# The selected mode is applied to every new reply
mode = st.radio("AI mode", list(MODES), horizontal=True)

# Chat history holds only user/assistant messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            system = SystemMessage(content=MODES[mode])
            response = get_llm().invoke([system] + st.session_state.messages)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))