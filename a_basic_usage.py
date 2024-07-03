from langchain_community.llms import Ollama

# 가장 기본적인 랭체인 사용법입니다.
# ollama로 llama3:8b를 설치한 뒤 진행하세요.

llm = Ollama(model="llama3:8b") # qwen2을 설치했다면, 해당 모델로 진행하세요.
result = llm.invoke("Tell me a joke")
print(result)
