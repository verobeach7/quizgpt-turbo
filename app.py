from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
import openai
import json
import os
import webbrowser


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


# API 키 유효성 검사 함수
def validate_openai_api_key(api_key):
    try:
        # OpenAI 라이브러리에 키 설정
        openai.api_key = api_key

        # 간단한 요청 보내기 (모델 목록 요청)
        openai.Model.list()
        return True  # 키가 유효함
    except openai.error.AuthenticationError:
        return False  # 키가 유효하지 않음
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False


# docs를 하나의 string으로 변경
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 프롬프트에 퀴즈 예시를 제공
questions_prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
            
    Based ONLY on the following context make 3 (THREE) questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    When creating questions, make sure to always take the difficulty level into 

    Context: {context}
    Difficulty: {difficulty}
    """
)


# file을 작은 chunk로 쪼갠 후 docs 반환
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def choose_difficulty():
    return st.selectbox(
        "Choose difficulty",
        (
            "High",
            "Low",
        ),
        index=None,
        placeholder="Select the difficulty",
    )


def main():
    st.markdown(
        """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
        """
    )
    if not openai_api_key:
        st.markdown("### Step 1. Get started by adding OpenAI API Key first.")
        return
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    @st.cache_data(show_spinner="Making quiz...")
    def run_quiz_chain(_docs, topic, difficulty):
        chain = (
            {
                "context": format_docs,
                "difficulty": lambda _: difficulty,
            }
            | questions_prompt
            | llm
        )
        return chain.invoke(_docs)

    if not docs:
        st.markdown(
            "### Step 2. And then uploading a file or searching on Wikipedia in the sidebar."
        )
    else:
        if not difficulty:
            st.markdown("### Step 3. Choose difficulty on the sidebar.")
        else:
            response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
            response = response.additional_kwargs["function_call"]["arguments"]
            questions = json.loads(response)["questions"]
            print(questions)
            with st.form("questions_form"):
                num_of_correct = 0
                for question in questions:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        num_of_correct += 1
                        st.success("Correct!")
                    elif value is not None:
                        st.error("Wrong!")
                st.form_submit_button()
                if len(questions) == num_of_correct:
                    st.balloons()


# 사이드바
with st.sidebar:
    docs = None
    topic = None
    url = "https://github.com/verobeach7/quizgpt-turbo/commit/0551d4756badce3a1c0f5a90ecf6bb7891a0039a"
    openai_api_key = st.text_input(
        "OpenAI_API_KEY", placeholder="Add your OpenAI API Key", type="password"
    )
    if openai_api_key:
        if validate_openai_api_key(openai_api_key):
            st.success("Your API Key is valid!")
        else:
            st.error("Invalid OpenAI API Key. Please check and try again.")
        # selectbox를 이용해 file/wikipedia article 중 선택
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        # File 선택
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
                difficulty = choose_difficulty()
        # Wikipedia Article 선택
        else:
            # 사용자로부터 topic 받기
            topic = st.text_input("Search Wikipedia...")
            if topic:
                # wiki_search를 이용하여 caching
                docs = wiki_search(topic)
                difficulty = choose_difficulty()
    st.link_button("GitHub Repo", url)

try:
    main()
except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)
