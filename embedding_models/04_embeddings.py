# Load the Hugging Face embedding model through LangChain, use the local sentence-transformers model to convert text into
# numerical vector representations, and generate embeddings for multiple sentences using embed_documents().

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model= "sentence-transformers/all-MiniLM-L6-v2"
)

# for single sentence
# vector = embeddings.embed_query("What is AI?")
# print(vector)

# for multiple sentences
texts = [
    "What is AI?",
    "What is ML",
    "What is DL",
    "What is NLP"
]

document_vectors = embeddings.embed_documents(texts)
print(document_vectors)


