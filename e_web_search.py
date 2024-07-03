import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 웹 페이지를 받아온 뒤, 해당 데이터를 llm에 넣어서
# 대답할 때 참조하도록 하는 예제입니다.

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "APIkey"

# 웹 문서 읽기 함수
def custom_web_loader(url):
    html_doc = requests.get(url)
    soup = BeautifulSoup(html_doc.text, 'html.parser')
    
    # <strong class="sa_text_strong"> 태그 내의 텍스트 추출
    text_elements = soup.find_all("strong", class_="sa_text_strong")
    text_list = [element.get_text(strip=True) for element in text_elements]
    
    return text_list

# 특정 URL에서 데이터 로드
url = "https://news.naver.com/section/105"
text_list = custom_web_loader(url)

# 리스트를 문자열로 변환하여 Document 객체 생성
documents = [Document(page_content=text, metadata={"source": url}) for text in text_list]

# OpenAI 임베딩 모델 로드
embeddings = OpenAIEmbeddings()

# 텍스트 데이터를 임베딩 벡터로 변환하여 벡터 db에 넣기
db = FAISS.from_documents(documents, embeddings)

# db 저장
db.save_local("faiss_index")

# 검색할 쿼리 문장
query = "AI software"

# 쿼리와 유사한 문서 검색
docs = db.similarity_search_with_score(query, k=3)  # 가장 유사한 문서 3개를 반환
for doc, score in docs:
    print(doc.page_content)
    print(f"Similarity score: {score}")


# 검색할 쿼리 문장
query = "우주개발과 AI"

# 쿼리와 유사한 문서 검색
docs = db.similarity_search_with_score(query, k=3)  # 가장 유사한 문서 3개를 반환
for doc, score in docs:
    print(doc.page_content)
    print(f"Similarity score: {score}")
