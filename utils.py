import streamlit as st
from langchain_openai import ChatOpenAI

def enable_chat_history(func):
    if "llm_ready" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_message(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm(temperature, max_tokens, top_p, openai_model, frequency, presence, openai_api_key):
    llm = ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model=openai_model,
        frequency_penalty=frequency,
        presence_penalty=presence,
        api_key=openai_api_key,
        streaming=True
    )

    return llm

def is_safe_query(query: str) -> bool:
    blocked_keywords = {"bypass", "jailbreak", "hack", "exploit", "password", "leak", "shutdown"}
    query_lower = query.lower()
    
    return (
        not any(keyword in query_lower for keyword in blocked_keywords) 
        and not detect_prompt_injection(query) 
        and len(query) < 200
    )

def detect_prompt_injection(query: str) -> bool:
    suspicious_phrases = {
        "ignore previous instructions", 
        "repeat after me", 
        "you are now"
    }
    
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in suspicious_phrases)

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
