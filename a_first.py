from langchain_community.llms import Ollama

# ollama serve할 필요 없습니다.

llm = Ollama(model="llama3:8b") # qwen2을 설치했다면, 해당 모델로 실행하세요.
result = llm.invoke("Tell me a joke") # 최신 버전에서는 invoke를 사용합니다.

print(result)
