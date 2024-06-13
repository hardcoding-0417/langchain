from langchain_community.embeddings import OpenVINOEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

# 사용할 모델 이름 지정
model_name = "sentence-transformers/all-mpnet-base-v2"

# 모델 매개변수 설정 (CPU에서 실행)
model_kwargs = {"device": "CPU"}

# 인코딩 매개변수 설정 (평균 풀링 및 임베딩 정규화 사용)
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

# OpenVINOEmbeddings를 사용하여 임베딩 모델 생성
ov_embeddings = OpenVINOEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# 테스트 문서
text = "This is a test document."

# 쿼리 임베딩 생성
query_result = ov_embeddings.embed_query(text)
print("Query embedding:", query_result[:3])

# 문서 임베딩 생성
doc_result = ov_embeddings.embed_documents([text])
print("Document embedding:", doc_result[0][:3])
