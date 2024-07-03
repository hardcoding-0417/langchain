from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

# 프롬프트와 랭체인 표현식 언어(LCEL)로 chain을 만드는 실습입니다.


# 좋은 프롬프트는 두 가지를 기본적으로 수행합니다.
# 1. AI의 역할 설정
# 2. 필요한 정보를 전달

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 사람들을 도와주는 착한 AI야."), # llm의 역할을 정합니다.
    ("human", "한국어로만 대화하자."),  # llm에게 내리는 첫 명령을 이와 같이 설정해놓을 수 있음.
    ("ai", "물론이죠! 한국어로만 대답하겠습니다!"), # 그에 대한 ai의 답변도 설정 가능.
    ("human", "{query}") # 유저의 인풋
])
llm = Ollama(model="qwen2")


# chain으로 묶어줍니다. 
chain = (
    {"query": RunnablePassthrough()} # 이 코드가 없다면 chain.invoke({"query": "who are you?})로 사용해야 합니다. 
    | prompt
    | llm
)

# |을 사용한 표현식 언어로 chain을 구성하는 것을 LCEL이라고 합니다.

# 기본적인 실행법
result = chain.invoke("hello")
print(result)

