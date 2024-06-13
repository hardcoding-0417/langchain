import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# ollama가 아니라 OpenAI를 쓴다면 api도 다뤄야 합니다.
# 윈도우 환경설정에 api 키를 등록해놓지 않았다면 아래의 코드를 사용해줍니다.

os.environ["OPENAI_API_KEY"] = "APIkey"
# 이처럼 api 키를 코드 상에 적어놓는 건 추천되지 않는 사용법입니다. 키가 털리면 api 요금으로 100만원이 날아올 수도 있으니까요.
# 그래서 깃허브 같은 데서는 API 키가 하드코딩 되어 있을 시 아예 push를 못하도록 막아놓습니다.


# streamlit을 사용해 간단한 이메일 작성기(웹 서비스)를 만들어보겠습니다.
# 실행은 streamlit run c_openai_and_streamlit.py로 합니다.

# 페이지 설정
st.set_page_config(page_title="이메일 전송 서비스입니다.", page_icon=":robot:")
st.header("이메일 전송기")

def getEmail():
    # 사용자로부터 이메일 입력 받기
    input_text = st.text_area(label="이메일 제목 입력", label_visibility='collapsed',
                              placeholder="당신의 제목을 입력하세요.", key="input_text")
    return input_text

# 이메일 입력값 받기
input_text = getEmail()

# 이메일 처리 준비
prompt = ChatPromptTemplate.from_template("{email} 제목의 이메일 내용을 작성해줘")
llm = ChatOpenAI(model_name="gpt-4", 
                 temperature=0.7, 
                 api_key=os.getenv("OPENAI_API_KEY"))
email_chain = prompt | llm # | StrOutputParser() # 출력을 확인했다면, 이번에는 파서를 추가해서 다시 해보세요.

# 이메일 제출 버튼
st.button("이메일 내용 생성", type='secondary', help="이메일 제목 입력을 완료했으면 이 버튼을 클릭하세요.")
st.markdown("### 생성된 이메일 내용:")

# 입력된 이메일이 있으면 처리
if input_text:
    result = email_chain.invoke({"email": input_text})
    st.write(result)

