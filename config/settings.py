import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_AI_MODEL: str = os.getenv('GROQ_AI_MODEL')
# print(GROQ_API_KEY)

model = ChatGroq(model=GROQ_AI_MODEL)

def get_model():
    return model