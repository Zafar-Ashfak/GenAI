import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
load_dotenv()


@st.cache_resource
def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        temperature=0.3,
        max_new_tokens=1000
    )
    return ChatHuggingFace(llm=llm)


class Movie(BaseModel):
    movie_name: str
    release_year: int
    release_date: Optional[str] = None
    genre: List[str]
    director: str
    producers: List[str]
    main_cast: List[str]
    characters: List[str]
    runtime: Optional[str]
    imdb_rating: str
    language: Optional[str]
    country: Optional[str]
    plot: Optional[str]
    themes: str
    quick_summary: str


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a movie information extraction assistant.

        Extract information from the movie paragraph.

        Return ONLY valid JSON.

        Do not return:
        - Markdown
        - ```json
        - explanations
        - additional text

        {format_instructions}
        """
    ),
    ("human", "{movie_text}")
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
            parser = PydanticOutputParser(pydantic_object=Movie)
            final_prompt = prompt.invoke({
                "movie_text": movie_text,
                "format_instructions": parser.get_format_instructions()
            })
            response = get_model().invoke(final_prompt)
            movie = parser.parse(response.content)
        st.json(movie.model_dump())