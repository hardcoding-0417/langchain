import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompt import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "api키 복붙"

# 사용할 모델 이름 지정
model_name = "sentence-transformers/all-mpnet-base-v2"

# 모델 매개변수 설정 (CPU에서 실행)
model_kwargs = {"device": "CPU"}

# 인코딩 매개변수 설정 (평균 풀링 및 임베딩 정규화 사용)
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

# OpenAIEmbeddings를 사용하여 임베딩 모델 생성
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
# 테스트 문서
text = "This is a test document."

# 쿼리 임베딩 생성
query_result = embeddings.embed_query(text)
print("Query embedding:", query_result[:3])

# 문서 임베딩 생성
doc_result = embeddings.embed_documents([text])
print("Document embedding:", doc_result[0][:3])
