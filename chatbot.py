import streamlit as st
import utils
from streaming import StreamHandler

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Chatbot")
st.title("🦜 Turing Chatbot")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai_model = st.sidebar.selectbox(
    "OpenAI Model",
    ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo-0125"),
)
max_tokens_model = 16384 if openai_model in {"gpt-4o", "gpt-4o-mini"} else 4096
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.1)
frequency = st.sidebar.slider("Frequency penalty", 0.0, 2.0, 0.0, 0.1)
presence = st.sidebar.slider("Presence penalty", 0.0, 2.0, 0.0, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 100, max_tokens_model, 2048, 100)

if openai_api_key.startswith("sk-"):
    st.session_state["llm_ready"] = True

class ContextChatbot:
    def __init__(self):
        utils.sync_st_session()
        self.llm = None

        if openai_api_key.startswith("sk-"):
            self.llm = utils.configure_llm(
                temperature,
                max_tokens,
                top_p,
                openai_model,
                frequency,
                presence,
                openai_api_key
            )
    
    @st.cache_resource
    def setup_chain(_self):    
        if _self.llm:
            memory = ConversationBufferMemory()
            chain = ConversationChain(llm=_self.llm, memory=memory, verbose=False)
            return chain
        return None
    
    @utils.enable_chat_history
    def main(self):
        if not self.llm:
            st.warning("Please provide a valid OpenAI API key to start chatting.", icon="⚠")
            return

        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Provide question!")
        if user_query:
            if not utils.is_safe_query(user_query):
                st.warning("🚫 This query is not allowed.", icon="⚠")
                return

            utils.display_message(user_query, "user")
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke({"input": user_query}, {"callbacks": [st_cb]})
                response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
