import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# print(GROQ_API_KEY)

model = ChatGroq(model='openai/gpt-oss-20b')
print(model)

# messages = [
#     SystemMessage(content="Give a me a random idiom with a keyword I give"),
#     HumanMessage(content="Sky")
# ]

# result = model.invoke(messages)

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
# parsed_output = parser.invoke(result)
# print(parsed_output)

# chain = model | parser
# print(chain.invoke(messages))


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [("system", "Give a me a random idiom with a keyword I give"),
     ("human", "{keyword}")]
)

chain = prompt | model | parser
print(chain.invoke({"keyword": "round"}))