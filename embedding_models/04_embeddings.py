from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "What is tokens?",
    "What is tokenization?",
    "What is an embedding?"
]

# query_vector = embeddings.embed_query("What is AI?")  # ---> Embed a single sentence
document_vectors = embeddings.embed_documents(texts) # ---> Embed multiple sentences

# print(query_vector)
print(document_vectors)
