# To load APIs from environment variable

from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("gemini-3.6-flash",
                        model_provider="google_genai")

response = model.invoke("What is Hugging face")

print(response.content)