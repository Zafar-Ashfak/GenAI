import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


@st.cache_resource
def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0.2,
        max_new_tokens=2048
    )
    return ChatHuggingFace(llm=llm)


prompt = ChatPromptTemplate([
    ("system", """
        You are a movie information extraction assistant.

        Your task is to analyze a movie paragraph and extract useful
        movie information such as movie name, release date, genre,
        director, producer, cast, actors, actresses, characters,
        runtime, IMDb rating, plot, themes, and a quick summary.

        Follow these rules:
        - Extract information only from the provided text.
        - Do not invent or assume missing information.
        - If information is missing, write "Not mentioned".
        - Keep the plot concise.
        - Generate a short 2-3 sentence quick summary.
        - Keep the output clear and structured.
    """),
    ("human", """
        Analyze the following movie paragraph:

        {movie_text}

        Extract the information in this format:

        Movie Name:
        Release Year:
        Release Date:
        Genre:
        Director:
        Producer:
        Main Cast:
        Actors:
        Actresses:
        Characters:
        Runtime:
        IMDb Rating:
        Language:
        Country:
        Plot:
        Themes:

        Quick Summary:
    """)
])


st.set_page_config(page_title="Movie Information Extractor")
st.title("Movie Information Extractor")

movie_text = st.text_area(
    "Movie paragraph",
    height=220,
    placeholder="Paste a paragraph about a movie here...",
)

if st.button("Extract information", type="primary"):
    if not movie_text.strip():
        st.warning("Enter a movie paragraph first.")
    else:
        with st.spinner("Extracting..."):
            final_prompt = prompt.invoke({"movie_text": movie_text})
            response = get_model().invoke(final_prompt)
        st.markdown(response.content)