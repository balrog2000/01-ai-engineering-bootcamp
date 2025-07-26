
import streamlit as st
from core.config import config
import requests
from typing import Dict, List, Any, cast
import json
from datetime import datetime
import uuid

st.set_page_config(page_title="Ecommerce assistant", layout="wide")

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id = get_session_id()

# Initialize debug data in session state
if 'debug_data' not in st.session_state:
    st.session_state.debug_data = []

def add_debug_entry(category: str, data: Any, title: str = ""):
    """Add an entry to the debug data"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {
        'timestamp': timestamp,
        'category': category,
        'title': title or category,
        'data': data
    }
    st.session_state.debug_data.append(entry)
    # Keep only last 20 debug entries
    if len(st.session_state.debug_data) > 20:
        st.session_state.debug_data.pop(0)

def api_call(method, url, **kwargs):
    def _show_error_popup(error_message): 
        st.session_state['error_popup'] = {
            'visible': True,
            'message': error_message
        }
        st.rerun()

    # Add debug entry for API call
    add_debug_entry("API Call", {
        'method': method,
        'url': url,
        'kwargs': kwargs
    }, f"API {method.upper()} Request")

    try:
        response = getattr(requests, method)(url, **kwargs)
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            response_data = {"message": "Error: Invalid JSON response"}

        # Add debug entry for API response
        add_debug_entry("API Response", {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'data': response_data
        }, f"API Response ({response.status_code})")

        if response.ok:
            return True, response_data
        return False, response_data

    except requests.exceptions.ConnectionError as e:
        add_debug_entry("API Error", {"error": "Connection error", "details": str(e)})
        _show_error_popup("Connection error")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout as e:
        add_debug_entry("API Error", {"error": "Timeout error", "details": str(e)})
        _show_error_popup("Timeout error")
        return False, {"message": "Timeout error"}
    except Exception as e:
        add_debug_entry("API Error", {"error": str(e), "type": type(e).__name__})
        _show_error_popup(f"Error: {e}")
        return False, {"message": f"Error: {e}"}

# Create two columns: chat and debug
chat_col, debug_col = st.columns([1, 1])

with chat_col:
    st.title("Ecommerce Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add embedding type selector, fusion checkbox, and chat input in columns
    col1, col2, col3 = st.columns([2, 1, 4])
    with col1:
        embedding_type = st.selectbox(
            "Embedding Type",
            options=[
                "Text embeddings (OpenAI)", 
                "Image embeddings (OpenCLIP)"
            ],
            key="embedding_type",
            label_visibility="collapsed",  
        )
    with col2:
        fusion = st.checkbox("Fusion", key="fusion", label_visibility="visible", value=True)
    with col3:
        prompt = st.chat_input("Hello! How can I help you today?")

    if prompt:
        # Add debug entry for user input
        # add_debug_entry("User Input", {"prompt": prompt, "timestamp": datetime.now().isoformat()})
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # output = run_llm(client, st.session_state.messages, max_tokens=st.session_state.max_tokens, temperature=st.session_state.temperature)
            api_embedding_type = embedding_type.lower().split(" ")[0]
            status, output = api_call('post', f'{config.API_URL}/rag', json={
                'query': prompt, 
                'embedding_type': api_embedding_type,
                'fusion': fusion,
                'thread_id': session_id
            })
            st.write(output['answer'])
            
            # Display items if available
            if 'items' in output and output['items']:
                st.markdown("### Related Products")
                # st.markdown(f"Used {output['used_context_count']} items initially found, not used {output['not_used_context_count']} items")
                
                # Create columns for the items
                cols = st.columns(min(len(output['items']), 6))  # Max 6 columns
                
                for idx, item in enumerate(output['items']):
                    col_idx = idx % 6
                    item_dict = cast(Dict[str, Any], item)
                    with cols[col_idx]:
                        with st.container():
                            st.markdown("---")
                            
                            # Display image if available
                            if item_dict.get('image_url'):
                                st.image(item_dict.get('image_url'), width=200, use_container_width=True)
                            
                            # Display description
                            if item_dict.get('description'):
                                st.markdown(f"**{item_dict.get('description')}**")
                            
                            # Display price if available
                            price = item_dict.get('price')
                            if price is not None:
                                st.markdown(f"💰 **${price:.2f}**")
                            else:
                                st.markdown("💰 **Price not available**")
            
        st.session_state.messages.append({"role": "assistant", "content": output['answer']})

# Debug Panel
with debug_col:
    st.title("🔧 Debug Panel")
    
    # # Debug controls
    # with st.expander("Debug Controls", expanded=True):
    #     if st.button("Clear Debug Data"):
    #         st.session_state.debug_data = []
    #         st.rerun()
        
    #     if st.button("Export Debug Data"):
    #         debug_json = json.dumps(st.session_state.debug_data, indent=2, default=str)
    #         st.download_button(
    #             label="Download Debug Data",
    #             data=debug_json,
    #             file_name=f"debug_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    #             mime="application/json"
    #         )
    
    # Session State Info
    with st.expander("Session State", expanded=False):
        session_info = {
            'message_count': len(st.session_state.messages),
            'debug_entries': len(st.session_state.debug_data),
            'session_keys': list(st.session_state.keys())
        }
        st.json(session_info)
    
    # Debug Entries
    with st.expander("Debug Entries", expanded=True):
        if not st.session_state.debug_data:
            st.info("No debug data yet. Start chatting to see debug information!")
        else:
            for entry in st.session_state.debug_data:  
                with st.container():
                    st.markdown(f"**{entry['timestamp']} - {entry['title']}**")
                    st.markdown(f"*Category: {entry['category']}*")
                    
                    # Display data based on type
                    if isinstance(entry['data'], dict):
                        st.json(entry['data'], expanded=1)
                    elif isinstance(entry['data'], (list, tuple)):
                        st.write(entry['data'])
                    else:
                        st.text(str(entry['data']))
                    st.markdown("---")    
    # API Configuration
    # with st.expander("API Configuration", expanded=False):
    #     st.json({
    #         'API_URL': config.API_URL,
    #         'config_keys': [key for key in dir(config) if not key.startswith('_')]
    #     })
