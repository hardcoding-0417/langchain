from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama


prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
output_parser = StrOutputParser()
llm = Ollama(model="llama3:8b")
chain = (
    {"topic": RunnablePassthrough()} 
    | prompt
    | llm
    | output_parser
)

result = chain.invoke("ice cream")

print(result)

