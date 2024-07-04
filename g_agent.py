from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os

# 랭체인에서는 agent라는 객체에 RAG에 필요한 기능들을 몰아넣어서 사용합니다.
# 매번 RAG를 만들지 않아도 되어서 매우 편리합니다.

# 에이전트 설정
memory = SqliteSaver.from_conn_string(":memory:")
llm = ChatOpenAI(model_name="gpt-4", 
                # ollama는 bind_tools이 없어서 사용이 불가능합니다.
                 temperature=0.7, 
                 api_key=os.getenv("OPENAI_API_KEY")) 
search = TavilySearchResults(max_results=2) # 검색 서비스입니다. api를 발급받아야 합니다.
tools = [search]
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# 에이전트 사용
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="안녕. 나는 미래고 학생이야.")]}, config
):
    print(chunk)

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="미래고산업고는 어떤 학교지?")]}, config
):
    print(chunk)

# 반환된 데이터를 적절히 파싱하여 사용하면 됩니다.
# API 사용량을 체크하고 싶다거나, 남들이 만들어놓은 에이전트를 가져다 쓰고 싶다면 랭스미스에 들러보세요.
