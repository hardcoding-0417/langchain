from langchain import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
import os

# 작업 설명
task_description = "작업: 주어진 단어의 반의어를 찾아주세요."

# 예시
examples = [
    {"word": "낮", "antonym": "밤"},
    {"word": "크다", "antonym": "작다"},
    {"word": "시작", "antonym": "끝"}
]

# 예시 템플릿
example_template = """
단어: {word}
반의어: {antonym}
"""

# 입력 변수
input_variables = ["word"]

# 예시 프롬프트 템플릿
example_prompt = PromptTemplate(
    input_variables=input_variables,
    template=example_template
)

# Few-shot 프롬프트 템플릿
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=task_description,
    suffix="단어: {input}\n반의어:",
    input_variables=["input"]
)

# OpenAI 언어 모델 초기화
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 입력 단어
word = "깨끗하다"

# Few-shot 프롬프트 생성
prompt = few_shot_prompt.format(input=word)

# 언어 모델을 사용하여 반의어 생성
response = llm.generate(prompt)

print(f"단어: {word}")
print(f"반의어: {response}")
