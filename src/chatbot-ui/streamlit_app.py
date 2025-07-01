import streamlit as st
from openai import OpenAI
from core.config import config
from google import genai
from groq import Groq

clients = {
    "openai": OpenAI(api_key=config.OPENAI_API_KEY),
    "google": genai.Client(api_key=config.GOOGLE_API_KEY),
    "groq": Groq(api_key=config.GROQ_API_KEY)
}

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

    provider = st.selectbox("Select a provider", provider_list)

    if provider is not None:
        model_name = st.selectbox('Model', model_lists[provider])
    
    st.session_state.provider = provider
    st.session_state.model_name = model_name

client = clients[st.session_state.provider]


def run_llm(client, messages, max_tokens=500):
    if st.session_state.provider == "google":
        return client.models.generate_content(
            model=st.session_state.model_name,
            contents=[message["content"] for message in messages],
        ).text
    else:
        return client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
            max_tokens=max_tokens
        ).choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = [
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
        output = run_llm(client, st.session_state.messages)
        st.write(output)
    st.session_state.messages.append({"role": "assistant", "content": output})