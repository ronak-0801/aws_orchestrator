import streamlit as st
import requests
import uuid
import json
from typing import Dict

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Configure the API URL
API_URL = "http://localhost:8000/chat/"

def stream_response(message: str) -> str:
    """Stream response from FastAPI backend"""
    try:
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        full_response = ""

        with requests.post(
            API_URL,
            json={
                "content": message,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id
            },
            stream=True
        ) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        # Decode the JSON response
                        response_data = json.loads(line)
                        if response_data.get("status") == "error":
                            return {"status": "error", "message": response_data.get("message", "Unknown error")}
                        
                        # Get the token
                        token = response_data.get("message", "")
                        print(token)
                        # full_response += token
                        # Update the placeholder with the accumulated response
                        response_placeholder.markdown(token + "▌")
                    except json.JSONDecodeError:
                        continue

        # Final update without the cursor
        response_placeholder.markdown(full_response)
        return {"status": "success", "message": full_response}

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the server: {str(e)}")
        return {"status": "error", "message": str(e)}

# Streamlit UI
st.title("Multi-Agent Chat System")

# Sidebar with session information
with st.sidebar:
    st.subheader("Session Information")
    st.write(f"User ID: {st.session_state.user_id}")
    st.write(f"Session ID: {st.session_state.session_id}")
    
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response with streaming
    with st.chat_message("assistant"):
        response = stream_response(prompt)
        if response["status"] == "success":
            st.session_state.messages.append({"role": "assistant", "content": response["message"]})
        else:
            st.error(f"Error: {response['message']}")

# Add a footer
st.markdown("---")
st.markdown("Powered by Isometrik Agents") 