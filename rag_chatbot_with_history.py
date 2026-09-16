from langchain import text_splitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from langchain_community.document_loaders import PyPDFLoader
model = settings.get_model()

loader = PyPDFLoader("assets/attention.pdf")
# print(loader.load())

session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

embeddings = settings.get_embeddings_hg()
txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = txt_splitter.split_documents(loader.load())

vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_store.as_retriever()

# print(retriever)
history = []
system_prompt_txt = ("You are a smart assistant who can help me based upon the context you've received. "
                 "Make sure you give single line answers and if you don't know the answer say 'I don't know' in one line. \n"
                     #"Also try to keep the conversation if it's something related to the context you've received or a normal chat question."
                     #" If the question is not related to the context, answer based upon your knowledge. \n"
                 "{context}")

prompt = ChatPromptTemplate([
    ("system", system_prompt_txt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]
)

history_aware_retriever = create_history_aware_retriever(model, retriever, prompt)
q_n_a_chain = create_stuff_documents_chain(model, prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, q_n_a_chain)

runnable_with_history = RunnableWithMessageHistory(rag_chain,
                                                   get_session_history,
                                                   history_messages_key="history",
                                                   input_messages_key="input",
                                                   output_messages_key="answer",)
res = runnable_with_history.invoke({"input": "What is self attention?"}, {"configurable": {"session_id": "chat_1"}})
print(res['answer'])
print('\n\n')

res2 = runnable_with_history.invoke({"input": "What are the different types of attentions?"}, {"configurable": {"session_id": "chat_1"}})
print(res2['answer'])
print('\n\n')

res3 = runnable_with_history.invoke({"input": "What are transformers?"}, {"configurable": {"session_id": "chat_1"}})
print(res3['answer'])
print('\n\n')

res4 = runnable_with_history.invoke({"input": "What is BERT?"}, {"configurable": {"session_id": "chat_1"}})
print(res4['answer'])
print('\n\n')


res5 = runnable_with_history.invoke({"input": "Give me a list of last 5 questions I asked?"}, {"configurable": {"session_id": "chat_1"}})
print(res5['answer'])
print('\n\n')
############ Retriever and chain without history #############
# q_n_a_chain = create_stuff_documents_chain(model, prompt)
#
# rag_chain = create_retrieval_chain(retriever, q_n_a_chain)
#
# response = rag_chain.invoke({"input": "What is self attention?"})
# print(response['answer'])