import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_AI_MODEL: str = os.getenv("GROQ_AI_MODEL")
HF_EMBEDDINGS_MODEL = os.getenv("HF_EMBEDDINGS_MODEL")
# print(GROQ_API_KEY)

model = ChatGroq(model=GROQ_AI_MODEL)


def get_model():
    return model

def get_embeddings_hg():
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)