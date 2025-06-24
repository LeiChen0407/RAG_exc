import streamlit as st
from rag_model_streamlit import RAGLegalSystem, load_rag_system
import os

# --- App Configuration ---
st.set_page_config(
    page_title="法律RAG问答系统",
    page_icon="⚖️",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Cached function to load the RAG system.
    It checks for config and index files and loads the system.
    If files are not found, it displays an error.
    """
    config_path = "rag_config.json"
    index_path = "legal_index.pkl"
    if not os.path.exists(config_path) or not os.path.exists(index_path):
        st.error(f"错误：找不到模型配置文件 ({config_path}) 或索引文件 ({index_path})。")
        st.error("请先运行 'python rag_model.py' 来初始化并保存RAG系统。")
        return None
    
    with st.spinner("正在加载RAG系统，请稍候..."):
        rag_system = load_rag_system(config_path=config_path, index_path=index_path)
    if rag_system:
        st.success("RAG系统加载成功！")
    return rag_system

rag_system = load_model()

# --- Session State Initialization ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [
        {"title": "新对话 1", "history": []}
    ]
if "current_chat_index" not in st.session_state:
    st.session_state.current_chat_index = 0

# --- Sidebar for Session Management ---
st.sidebar.title("聊天会话")

if st.sidebar.button("➕ 新建对话"):
    session_num = len(st.session_state.chat_sessions) + 1
    st.session_state.chat_sessions.append(
        {"title": f"新对话 {session_num}", "history": []}
    )
    st.session_state.current_chat_index = len(st.session_state.chat_sessions) - 1
    st.rerun()

session_titles = [s["title"] for s in st.session_state.chat_sessions]
st.session_state.current_chat_index = st.sidebar.radio(
    "选择一个对话:",
    range(len(session_titles)),
    format_func=lambda i: session_titles[i],
    index=st.session_state.current_chat_index
)

st.sidebar.info("您可以新建多个对话。每个对话的上下文都是独立的。")


# --- Main Chat Interface ---
current_session = st.session_state.chat_sessions[st.session_state.current_chat_index]
st.header(f"对话: {current_session['title']}")

# Display chat history
for message in current_session["history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("请输入您的问题..."):
    if rag_system is None:
        st.error("RAG系统未加载，无法回答问题。")
    else:
        # Add user message to history and display it
        current_session["history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # If this is the first message, update the session title
        if len(current_session["history"]) == 1:
            # Generate a title from the first prompt
            new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            st.session_state.chat_sessions[st.session_state.current_chat_index]["title"] = new_title

        # Generate and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = rag_system.generate_conversational_answer(
                    question=prompt,
                    chat_history=current_session["history"][:-1] # Exclude current prompt from history for generation
                )
                st.markdown(response)
        
        # Add assistant response to history
        current_session["history"].append({"role": "assistant", "content": response})

if not rag_system:
    st.warning("系统未就绪，请先确保 `rag_model.py` 已成功运行。") 