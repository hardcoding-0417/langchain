import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from PIL import Image

# OpenAI의 API를 받아 streamlit으로 간단한 서비스를 만들어보는 예제입니다.
# 환경설정에 api 키를 등록하세요.

# 만약 환경설정에 등록하기 싫다면, 아래의 코드를 사용합니다.

# os.environ["OPENAI_API_KEY"] = "APIkey"
# 주의. 이처럼 api 키를 코드 상에 적어놓는 건 추천되지 않는 사용법입니다.
# 키가 털리면 api 요금으로 100만원이 날아올 수도 있으니까요.
# (그래서 깃허브 같은 데서는 API 키가 하드코딩 되어 있을 시 아예 push를 못하도록 막아놓습니다.)


# streamlit은 복잡한 웹 지식 없이도
# 쉽고 간단하게 웹 어플리케이션을 만들 수 있게 해주는 프레임워크입니다.

# 페이지 설정
st.set_page_config(page_title="이메일 전송 서비스입니다.", page_icon=":robot:", layout="wide") # layout의 인자를 centereds나 wide로 다르게 해보세요.
st.header("이메일 전송기")

# 대문 이미지 띄우기
image = Image.open("header_image.jpg")
st.image(image, use_column_width=True)

# 사용자로부터 이메일 입력 받기
def getEmail():
    input_text = st.text_area(label="이메일 제목 입력", label_visibility='collapsed',
                              placeholder="이메일 제목을 입력하세요.", key="input_text")
    return input_text

# 이메일 입력값 받기
input_text = getEmail()

# 이메일 처리 준비
prompt = ChatPromptTemplate.from_template("{email} 라는 제목의 이메일을 쓸 거야. 안에 들어갈 내용을 작성해줘")
llm = ChatOpenAI(model_name="gpt-4", 
                 temperature=0.7, 
                 api_key=os.getenv("OPENAI_API_KEY"))
email_chain = prompt | llm | StrOutputParser() # 출력을 확인했다면, 이번에는 파서를 없애고 다시 해보세요.

# 이메일 제출 버튼
if st.button("이메일 내용 생성", type='secondary', help="이메일 제목 입력을 완료했으면 이 버튼을 클릭하세요."):
    # 입력된 이메일이 있으면 처리
    if input_text:
        result = email_chain.invoke({"email": input_text})
        
        col1, col2 = st.columns([2,1]) # 단순히 숫자를 줄 시 화면을 동등한 비율로 나누고, 2,1로 주면 2:1로 나눕니다.
        with col1:
            st.markdown("### 생성된 이메일 내용:")
            st.write(result)

        with col2:
            st.markdown("### 이미지:")
            image = Image.open("side_image.jpg")
            st.image(image, use_column_width=True)

# 실행은 streamlit run c_openai_and_streamlit.py로 합니다.
