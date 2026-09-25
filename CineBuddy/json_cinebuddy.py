from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation",
        temperature=0.3,
        max_new_tokens=1000
    )

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
    imdb_rating: Optional[float]
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

def main():
    movie_text = input("Enter movie text:\n")
    parser = PydanticOutputParser(pydantic_object=Movie)

    final_prompt = prompt.invoke({
        "movie_text": movie_text,
        "format_instructions": parser.get_format_instructions()
    })

    llm = get_llm()
    model = ChatHuggingFace(llm=llm)
    response = model.invoke(final_prompt)
    movie_json = parser.parse(response.content)
    print(movie_json.model_dump_json(indent=2))

main()

