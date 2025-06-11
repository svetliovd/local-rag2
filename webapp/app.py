import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_ollama import OllamaLLM as Ollama
from transformers import GenerationConfig
import os
import pymupdf4llm
import tempfile
import pandas as pd
import docx
import torch
import gc

# Почистване на meta tensor проблема
torch.classes.__path__ = []

CHROMA_DIR = "./chroma_db"

# Настройки
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

st.set_page_config(
    page_title="Вашият AI асистент",
    page_icon="./favicon.png"
)

# Странични контроли
st.sidebar.header("⚙️ Настройки")
language = st.sidebar.selectbox("Избери език за обработка:", ["български", "английски"])

# Функции за зареждане на модели и векторна база
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

@st.cache_resource(show_spinner=False)
def get_text_splitter():
    return SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size=100,
        chunk_overlap=20
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    generation_params = {
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "top_k": 25,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "eos_token_id": [1, 107],
        "do_sample": True
    }

    return Ollama(
        base_url=ollama_host,
        model="todorov/bggpt:9B-IT-v1.0.Q8_0",
        generation_config=generation_params
    )

def clear_vector_store():
    import shutil
    import os

    # Изтриване на директорията
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    # Пресъздаване на празна директория с правилни права
    os.makedirs(CHROMA_DIR, exist_ok=True)
    # Изчистване на кешовете

@st.cache_resource(show_spinner=False)
def get_vector_store():
    return Chroma(
        embedding_function=get_embedding_model(),
        persist_directory="./chroma_db",  # локалната папка
        collection_name="default"
    )

# Зареждане на документи
@st.cache_data(show_spinner=False)
def load_documents(uploaded_files):
    text_splitter = get_text_splitter()
    extracted_pages = []
    documents = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(file.read())
                tmp_pdf.flush()
                pdf_text = pymupdf4llm.to_markdown(tmp_pdf.name)
                documents.append(pdf_text)
                extracted_pages.append((file.name, [pdf_text]))
        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
            documents.append(text)
            extracted_pages.append((file.name, [text]))
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_string(index=False)
            documents.append(text)
            extracted_pages.append((file.name, [text]))
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
            text = df.to_string(index=False)
            documents.append(text)
            extracted_pages.append((file.name, [text]))
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            documents.append(text)
            extracted_pages.append((file.name, [text]))
        else:
            st.warning(f"Неподдържан тип файл: {file.name}")

    chunks = text_splitter.split_text("\n".join(documents))
    get_vector_store().add_texts(chunks)

    return extracted_pages

# CSS стилове
custom_css = """
<style>
.stApp { background-color: #1C1C1C; }
.stApp, .stMarkdown, .stText, .stTextArea, .stRadio > label, .stButton > button {
    color: #F5E8D8;
}
.css-1v0mbdj, .css-17eq0hr, .css-1v3fvcr { color: #F5E8D8 !important; }
.stButton > button {
    background-color: #3E5641;
    color: #F5E8D8;
}
.stExpander, .stExpander div { color: #F5E8D8; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Инициализация на чат историята
if "history" not in st.session_state:
    st.session_state.history = []

# Основен интерфейс
st.title("Вашият AI асистент")

# Качване на файлове
uploaded_files = st.file_uploader(
    "Качи документи (PDF, TXT, XLSX, CSV, DOCX)",
    accept_multiple_files=True,
    type=["txt", "pdf", "xlsx", "csv", "docx"]
)

if uploaded_files:
    extracted_pages = load_documents(uploaded_files)
    st.success("Документите са заредени успешно!")
    st.subheader("📄 Извлечено съдържание от документите:")
    for filename, pages in extracted_pages:
        with st.expander(f"{filename} ({len(pages)} стр.)"):
            tab_names = [f"Страница {i + 1}" for i in range(len(pages))]
            tabs = st.tabs(tab_names)
            for tab, page in zip(tabs, pages):
                with tab:
                    page_content = page if isinstance(page, str) else page.page_content
                    st.write(page_content)

# Секция за чат историята
st.subheader("🗨️ История на чата")

# Бутон за изчистване на чат историята
if st.button("🗑️ Изчисти историята на чата", key="clear_chat_history"):
    st.session_state.clear()  # Clears all session data including history
    st.rerun()  # Immediately refreshes the interface
    
# Бутон за изчистване на векторната база
with st.sidebar.expander("🧹 Изчисти векторната база"):
    st.warning("⚠️ Това ще изтрие всички документи от векторната база и не може да бъде върнато обратно.")
    if st.sidebar.button("✅ Потвърди изчистване", key="confirm_clear_vector_db"):
        clear_vector_store()
        st.success("Векторната база е изчистена успешно!")
        st.rerun()

# Показване на чат историята
chat_container = st.container()
with chat_container:
    if st.session_state.history:
        for i, (question, answer) in enumerate(st.session_state.history):
            with st.expander(f"Въпрос {i + 1}: {question[:50]}..."):
                st.markdown(f"**Въпрос:** {question}")
                st.markdown(f"**Отговор:** {answer}")
    else:
        st.info("Няма запазена история на чата.")

# Форма за въпроси
st.subheader("Задай въпрос")
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_area("Въведи твоя въпрос тук:", height=150, key="query_input")
    submit_button = st.form_submit_button("Попитай")

# Обработка на въпроса
if submit_button and query:
    docs = get_vector_store().similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    chat_history = ""
    for i, (past_query, past_response) in enumerate(st.session_state.history[-5:]):
        chat_history += f"Въпрос {i+1}: {past_query}\nОтговор {i+1}: {past_response}\n\n"

    lang_text = "български" if language == "български" else "английски"

    prompt = (
        f"История на чата:\n{chat_history}\n"
        f"Контекст от документи: {context}\n"
        f"Нов въпрос: {query}\n"
        f"Отговор на {lang_text}:"
    )

    llm = get_llm()
    st.write("**Отговор:**")
    response_container = st.empty()
    full_response = ""

    try:
        for chunk in llm.stream(prompt):
            full_response += chunk
            response_container.markdown(full_response)
    except AttributeError:
        response = llm.invoke(prompt)
        full_response = response
        response_container.markdown(full_response)

    st.session_state.history.append((query, full_response))
    gc.collect()
