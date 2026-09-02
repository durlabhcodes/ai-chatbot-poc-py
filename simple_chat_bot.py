from config import settings

model = settings.get_model()

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
    [("system", "Give a me a random idiom with a keyword I give"), ("human", "{keyword}")]
)

chain = prompt | model | parser
print(chain.invoke({"keyword": "round"}))
