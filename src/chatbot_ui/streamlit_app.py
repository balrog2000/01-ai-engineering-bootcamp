from qdrant_client import QdrantClient
import streamlit as st
from openai import OpenAI
from core.config import config
from google import genai
from groq import Groq
from google.genai.types import GenerateContentConfig
from retrieval import rag_pipeline

clients = {
    "openai": OpenAI(api_key=config.OPENAI_API_KEY),
    "google": genai.Client(api_key=config.GOOGLE_API_KEY),
    "groq": Groq(api_key=config.GROQ_API_KEY)
}

qdrant_client = QdrantClient(
    url=f"http://{config.QDRANT_HOST}:6333",
)

st.title("Chatbot UI")

## lets create a sidebar with a dropdown for the model list and providers
model_lists = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4o-turbo", "gpt-4o-turbo-preview", "gpt-4o-turbo-preview-2024-07-18"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-lite-preview", "gemini-2.0-flash-lite-preview-2024-07-18"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.3-70b-versatile-preview", "llama-3.3-70b-versatile-preview-2024-07-18"]
}
provider_list = ["openai", "google", "groq"]

with st.sidebar:
    st.title("Settings")

    provider = st.selectbox("Select a provider!", provider_list)

    if provider is not None:
        model_name = st.selectbox('Model', model_lists[provider])
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5, step=1)
    max_tokens = st.number_input("Max Tokens", min_value=1, value=500, step=200)
    
    st.session_state.provider = provider
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens
    st.session_state.top_k = top_k

client = clients[st.session_state.provider]


def run_llm(client, messages, max_tokens=500, temperature=1.0):
    if st.session_state.provider == "google":
        return client.models.generate_content(
            model=st.session_state.model_name,
            contents=[message["content"] for message in messages],
            config=GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        ).text
    else:
        # seems the max_tokens is now deprecated
        return client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature
        ).choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You should never disclose what model are you based on!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]




for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # output = run_llm(client, st.session_state.messages, max_tokens=st.session_state.max_tokens, temperature=st.session_state.temperature)
        output = rag_pipeline(prompt, qdrant_client, top_k=st.session_state.top_k)
        st.write(output['answer'])
    st.session_state.messages.append({"role": "assistant", "content": output['answer']})