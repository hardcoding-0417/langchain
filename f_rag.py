import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import os

# RAG란 정보를 검색한 다음, 해당 정보를 참조해서 답변하도록 하는 기술입니다.
# 파인튜닝 없이도 곧바로 성능을 향상시킬 수 있기 때문에 매우 유용합니다.

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "API키를 입력하세요"

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    # 곧바로 bs4를 써서 파싱하는 예시 입니다.
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# chroma도 벡터 db 중 하나입니다.
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

llm = Ollama(model="qwen2")

# 검색을 통해 정보를 얻고, 해당 정보를 참조해서 답변을 생성하도록 합니다.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt") # 프롬프트를 직접 만드는 대신, 랭 스미스의 허브에서 받아옵니다.
# 해당 프롬프트에 대한 자세한 정보는 다음의 링크에 있습니다. https://smith.langchain.com/hub/rlm/rag-prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is Task Decomposition?")
print (result)