
import streamlit as st
from core.config import config
import requests
from typing import Dict, List, Any

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
        _show_error_popup("Connection error")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout as e:
        _show_error_popup("Timeout error")
        return False, {"message": "Timeout error"}
    except Exception as e:
        _show_error_popup(f"Error: {e}")
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
        
        # Display items if available
        if 'items' in output and output['items']:
            st.markdown("### Related Products")
            
            # Create columns for the items
            cols = st.columns(min(len(output['items']), 3))  # Max 3 columns
            
            for idx, item in enumerate(output['items']):
                col_idx = idx % 3
                with cols[col_idx]:
                    with st.container():
                        st.markdown("---")
                        
                        # Display image if available
                        if 'image_url' in item and item['image_url']:
                            st.image(item['image_url'], width=200, use_container_width=True)
                        
                        # Display description
                        if 'description' in item and item['description']:
                            st.markdown(f"**{item['description']}**")
                        
                        # Display price if available
                        if 'price' in item:
                            if item['price'] is not None:
                                st.markdown(f"💰 **${item['price']:.2f}**")
                            else:
                                st.markdown("💰 **Price not available**")
                        else:
                            st.markdown("💰 **Price not available**")
        
    st.session_state.messages.append({"role": "assistant", "content": output['answer']})