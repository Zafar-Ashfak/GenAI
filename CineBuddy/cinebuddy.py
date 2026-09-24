from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0.2
    )

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

def main():
    llm = get_llm()
    model = ChatHuggingFace(llm=llm)
    movie_text = input("Enter a movie information\n")
    final_prompt = prompt.invoke({
            "movie_text": movie_text
        }
    )

    response = model.invoke(final_prompt)

    print(response.content)

main()