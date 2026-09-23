import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Your job is to answer the user's questions clearly, accurately, and in simple language.
Give direct answers and avoid unnecessary information.
For technical questions, explain concepts with simple examples when useful.
If you are unsure about something, clearly say that you are unsure instead of making up information.
Maintain a friendly and professional tone.
"""

@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


st.title("Meta Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# Show past messages (skip the system prompt)
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(m.content)

if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm().invoke(st.session_state.messages)
        st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))