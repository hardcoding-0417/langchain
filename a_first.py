from langchain_community.llms import Ollama

llm = Ollama(model="llama3:8b")

result = llm.invoke("Tell me a joke") # langchain 문서를 보면invoke는

print(result)