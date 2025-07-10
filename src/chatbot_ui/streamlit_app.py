
import streamlit as st
from core.config import config
import requests

st.set_page_config(page_title="Ecommerce assistant", layout="wide")



def api_call(method, url, **kwargs):
    def _show_error_popup(error_message): 
        st.session_state['error_popup'] = {
            'visible': True,
            'message': error_message
        }
        st.rerun()


    try:
        response = getattr(requests, method)(url, **kwargs)
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            response_data = {"message": "Error: Invalid JSON response"}

        if response.ok:
            return True, response_data
        return False, response_data

    except requests.exceptions.ConnectionError as e:
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout as e:
        return False, {"message": "Timeout error"}
    except Exception as e:
        return False, {"message": f"Error: {e}"}

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # output = run_llm(client, st.session_state.messages, max_tokens=st.session_state.max_tokens, temperature=st.session_state.temperature)
        status, output = api_call('post', f'{config.API_URL}/rag', json={'query': prompt})
        st.write(output['answer'])
    st.session_state.messages.append({"role": "assistant", "content": output['answer']})