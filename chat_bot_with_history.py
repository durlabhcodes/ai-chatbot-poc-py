from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

from config import settings

model = settings.get_model()
parser = StrOutputParser()

session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


history = []
prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a wise genius who gives advice to humans related to the topic they provide. "
            "You will be given a topic and you will provide advice related to that topic in one single line."
            "Also if the human asks a normal question you will answer based upon the history in one line. "
            "If you don't know the answer you will say 'I don't know' in one line. ",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


chain = RunnablePassthrough.assign(input_length=lambda x: len(x)) | prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history, history_messages_key="history", input_messages_key="input"
)

response = chain_with_history.invoke(
    {"input": "Global Warming"}, {"configurable": {"session_id": "chat_1"}}
)

print("First Response with RunnableWithMessageHistory ===>> " + str(response))

response = chain_with_history.invoke(
    {"input": "What was our last topic?"}, {"configurable": {"session_id": "chat_1"}}
)

print("Second Response with RunnableWithMessageHistory ===>> " + str(response))

## This is how you can use the chain without using RunnableWithMessageHistory class and manually create the history

# response = chain.invoke({"history": history, "input":"Artificial Intelligence"})
#
# # Manual History Creation without using the InMemoryChatMessageHistory class
# history.append(HumanMessage(content = "Artificial Intelligence"))
# history.append(response)
# print("First Response with no history ===>> " + str(response.content))
#
# response = chain.invoke({"history": history, "input": "What was the last topic I asked you about?"})
# print("Second Response with no history ===>> " + str(response.content))
#
# # Manual History Creation without using the InMemoryChatMessageHistory class
# history.append(HumanMessage(content = "What was the last topic I asked you about?"))
# history.append(response)
# print("History at first checkpoint ===>> "+str(history))
