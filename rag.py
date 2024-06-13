from langchain.vectorstores import FAISS
import os
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
)


# OpenVINO 임베딩 모델 로드
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "CPU"}
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

# 임베딩 생성
embeddings = OpenVINOEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# FAISS 벡터 데이터베이스 로드
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 기본 템플릿 및 모델 설정
prompt_template = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
{context}
Question: {question}
""")
llm = Ollama(model="llama3:8b")

# 검색 리트리버 구성
retriever = db.as_retriever()
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
    id="search_kwargs", 
    name="Search Kwargs", 
    description="The search kwargs to use"
))

# 체인 구성
chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

