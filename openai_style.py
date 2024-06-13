import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompt import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = "api키 복붙"

# 페이지 설정
st.set_page_config(page_title="이메일 전송 서비스입니다.", page_icon=":robot:")
st.header("이메일 전송기")

def getEmail():
    # 사용자로부터 이메일 입력 받기
    input_text = st.text_area(label="이메일 입력", label_visibility='collapsed',
                              placeholder="당신의 이메일을 제출은...", key="input_text")
    return input_text

# 이메일 입력값 받기
input_text = getEmail()

# 이메일 처리 준비
prompt = ChatPromptTemplate.from_template("이메일 제목에 맞는 내용을 작성해줘")
llm = ChatOpenAI(model_name="gpt-4", 
                 temperature=0.7, 
                 api_key=os.getenv("OPENAI_API_KEY"))
email_chain = prompt | llm | StrOutputParser()

# 이메일 제출 버튼
st.button("이메일 제목을 작성하세요", type='secondary', help="좋은 제목한 이메일을 확인해보세요.")
st.markdown("### 본인이 작성한 이메일 내용은:")

# 입력된 이메일이 있으면 처리
if input_text:
    result = email_chain.invoke({"email": input_text})
    st.write(result)
