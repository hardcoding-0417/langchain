import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 벡터 DB를 만들고 저장하는 예제입니다.
# 유사도 검색 대신 llm을 사용하면 RAG가 됩니다.

# OpenAI API 키 설정 (환경 변수를 통해 설정)
os.environ["OPENAI_API_KEY"] = "APIkey"

pdf_path = '구운몽.pdf'

# PDF 파일 읽기
with open(pdf_path, 'rb') as file:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)
texts = text_splitter.split_text(text)
print(f"Number of text chunks: {len(texts)}")

# 임베딩 생성
embeddings = OpenAIEmbeddings()

# 벡터 데이터베이스에 저장
docsearch = FAISS.from_texts(texts, embeddings)

# 유사도 검색
query = "양소유는 누구?"
docs = docsearch.similarity_search_with_score(query, k=3) 
# 가장 유사한 것 4개를 반환하는 메서드. 하나만 반환하고 싶으면 k=1을 사용

for doc, score in docs:
    print(f"Similarity score: {score}")
    print(doc.page_content)
    print()