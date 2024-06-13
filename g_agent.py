# Import relevant functionality
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os


os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["TAVILY_API_KEY"] = "TAVILY_API_KEY"

# 에이전트 설정
memory = SqliteSaver.from_conn_string(":memory:")
llm = ChatOpenAI(model_name="gpt-4", 
                 temperature=0.7, 
                 api_key=os.getenv("OPENAI_API_KEY")) # ollama에는 bind_tools이 없어서 사용이 불가능합니다.
search = TavilySearchResults(max_results=2) # 검색 서비스입니다. api를 발급받아야 합니다.
tools = [search]
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# 에이전트 사용
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="안녕. 나는 세명고 학생이야.")]}, config
):
    print(chunk)
    print("----")

    # 각 청크가 ----로 분리되어 있습니다. 이것을 단계별로 잘 살펴보면 에이전트가 어떤 단계를 거쳐서 정보를 찾아냈는지 손쉽게 알 수 있습니다.

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="세명 컴퓨터 고등학교는 어떤 학교지?")]}, config
):
    print(chunk)
    print("----")

# 뭔가 복잡한 결과가 나올 텐데, 해당 결과는 JSON데이터입니다.
# gpt에게 json형식에 맞춰 보기쉽게 정리해달라고 하면 아래와 같이 정리해줍니다.
# 적절히 파싱하면 사용자에게 원하는 정보를 전달할 수 있을 것입니다.

"""

[
    {
        "agent": {
            "messages": [
                {
                    "content": "안녕하세요! 세명고 학생님, 어떻게 도와드릴 수 있을까요?",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 36,
                            "prompt_tokens": 98,
                            "total_tokens": 134
                        },
                        "model_name": "gpt-4",
                        "system_fingerprint": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "id": "run-f5b7a7a4-fa40-49cb-876d-74c57f228721-0",
                    "usage_metadata": {
                        "input_tokens": 98,
                        "output_tokens": 36,
                        "total_tokens": 134
                    }
                }
            ]
        }
    },
    {
        "agent": {
            "messages": [
                {
                    "content": "",
                    "additional_kwargs": {
                        "tool_calls": [
                            {
                                "id": "call_Jknba6z1w0scV30vOaQ0l3MP",
                                "function": {
                                    "arguments": "{\n  \"query\": \"세명 컴퓨터 고등학교\"\n}",
                                    "name": "tavily_search_results_json"
                                },
                                "type": "function"
                            }
                        ]
                    },
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 34,
                            "prompt_tokens": 169,
                            "total_tokens": 203
                        },
                        "model_name": "gpt-4",
                        "system_fingerprint": null,
                        "finish_reason": "tool_calls",
                        "logprobs": null
                    },
                    "id": "run-d70d120d-1125-427a-a858-a71d5575e22b-0",
                    "tool_calls": [
                        {
                            "name": "tavily_search_results_json",
                            "args": {
                                "query": "세명 컴퓨터 고등학교"
                            },
                            "id": "call_Jknba6z1w0scV30vOaQ0l3MP"
                        }
                    ],
                    "usage_metadata": {
                        "input_tokens": 169,
                        "output_tokens": 34,
                        "total_tokens": 203
                    }
                }
            ]
        }
    },
    {
        "tools": {
            "messages": [
                {
                    "content": "[{\"url\": \"https://m.blog.naver.com/urmyver/222836549931\", \"content\": \"\\uc11c\\uc6b8\\uc5d0 \\uc18c\\uc7ac\\ud55c it \\ud2b9\\uc131\\ud654\\uace0\\ub4f1\\ud559\\uad50 \\uc911 \\ud558\\ub098\\uc778 \\uc138\\uba85\\ucef4\\ud4e8\\ud130\\uace0\\ub4f1\\ud559\\uad50\\ub294 \\uc8fc\\ub85c \\uc77c\\ucc0d\\uc774 \\uc9c4\\ub85c\\ub97c it \\ubd84\\uc57c\\ub85c \\uc124\\uc815\\ud55c \\uc911\\ud559\\uc0dd\\ub4e4\\uc774 \\uc9c4\\ud559\\uc744 \\uace0\\ub824\\ud558\\ub294 \\ud559\\uad50\\ub85c \\ub2e4\\ub978 \\ud2b9\\uc131\\ud654\\uace0\\ub4f1\\ud559\\uad50\\uc5d0 \\ube44\\ud558\\uc5ec \\uc878\\uc5c5\\ud55c \\ud559\\uc0dd\\ub4e4\\uc758 4\\ub144\\uc81c\\uc778 \\uc11c\\uc6b8\\uad8c \\ub300\\ud559 \\uc9c4\\ud559\\ub960\\uc774 \\uc0c1\\ub2f9\\ud788 \\ub192\\ub2e4 \\ub294 \\uc7a5\\uc810\\uc774 \\uc788\\uc5b4\\uc11c it \\ubd84\\uc57c\\uc640 \\uc801\\uc131\\uc774 \\uc798 \\ub4e4\\uc5b4\\ub9de\\uc744 \\ub54c \\uc9c4\\ud559\\ud558\\uae30 ...\"}, {\"url\": \"https://namu.wiki/w/\\uc138\\uba85\\ucef4\\ud4e8\\ud130\\uace0\\ub4f1\\ud559\\uad50\", \"content\": \"\\uc11c\\uc6b8\\uc5d0\\uc11c AI, \\ubcf4\\uc548, \\uac8c\\uc784 \\uc774\\ub77c \\ud558\\uba74 \'\\uc138\\uba85\\ucef4\\ud4e8\\ud130\\uace0\\ub4f1\\ud559\\uad50\'\\ub77c\\uace0 \\ud560 \\uc218 \\uc788\\uc744 \\uc815\\ub3c4\\ub85c [36] \\uc720\\uba85\\ud55c \\uad6d\\ub0b4 \\ucd5c\\uace0 IT\\uba85\\ubb38\\ud559\\uad50\\uc774\\ub2e4 \\ucd5c\\uadfc\\uc5d4 \\uc880 \\ub354 \\uccb4\\uacc4\\uc801\\uc778 \\uae30\\uc220\\uc744 \\ud65c\\uc6a9\\ud558\\uc5ec VR,AR,AI\\ub4f1 4\\ucc28\\uc0b0\\uc5b5\\ud601\\uba85 \\uc2dc\\ub300\\uc5d0 \\ubc1c\\ub9de\\ucdb0 \\uad50\\uc721\\uc744 \\uc2e4\\ud589\\ud558\\uace0 \\uc788\\ub2e4. \\uad6d\\ub0b4 \\ucd5c\\ucd08 \'\\uc18c\\ud504\\ud2b8\\uc6e8\\uc5b4 \\uc120\\ub3c4\\ud559\\uad50 \'\\uc774\\uc790 ...\"}]",
                    "name": "tavily_search_results_json",
                    "tool_call_id": "call_Jknba6z1w0scV30vOaQ0l3MP"
                }
            ]
        }
    },
    {
        "agent": {
            "messages": [
                {
                    "content": "세명 컴퓨터 고등학교는 서울에 위치한 IT 특성화 고등학교 중 하나입니다. 주로 일반적인 진로를 IT 분야로 설정한 중학생들이 진학을 고려하는 학교로, 다른 특성화 고등학교에 비해 졸업한 학생들의 4년제 서울권 대학 진학률이 상당히 높은 편입니다. [출처](https://m.blog.naver.com/urmyver/222836549931)\n\n세명 컴퓨터 고등학교는 AI, 보안, 게임 등의 분야에서 유명하며, 국내 최고의 IT 명문 학교로 알려져 있습니다. 최근에는 VR, AR, AI 등 4차산업혁명 시대에 발맞춰 교육을 실시하고 있으며, 국내 최초 '소프트웨어 선도학교'로서의 역할을 하고 있습니다. [출처](https://namu.wiki/w/세명컴퓨터고등학교)",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 341,
                            "prompt_tokens": 1050,
                            "total_tokens": 1391
                        },
                        "model_name": "gpt-4",
                        "system_fingerprint": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "id": "run-3f2a0282-7e96-4042-9f03-1a74213ba61c-0",
                    "usage_metadata": {
                        "input_tokens": 1050,
                        "output_tokens": 341,
                        "total_tokens": 1391
                    }
                }
            ]
        }
    }
]
"""