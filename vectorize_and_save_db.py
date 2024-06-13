import os
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader


# 웹 문서 읽기
loader = WebBaseLoader("https://www.espn.com/")
data = loader.load()

# OpenVINO 임베딩 모델 로드
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "CPU"}
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}
embeddings = OpenVINOEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# 텍스트 데이터를 임베딩 벡터로 변환하여 db에 넣기
db = FAISS.from_documents(data, embeddings)

# db 저장
db.save_local("faiss_index")

# 검색할 쿼리 문장
query = "What's the latest news in sports?"

# 쿼리와 유사한 문서 검색
docs = db.similarity_search(query)
# 검색 결과 출력
for doc in docs:
    print(docs[0].page_content)