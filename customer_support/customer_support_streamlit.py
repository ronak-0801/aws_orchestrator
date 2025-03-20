import streamlit as st
import asyncio
import time
from customer_support import process_customer_query, DEFAULT_USER
import json
from datetime import datetime
import threading
from queue import Queue

# Set page configuration
st.set_page_config(
    page_title="TechGadgets Customer Support",
    page_icon="💬",
    layout="centered"
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Display header
st.title("TechGadgets Customer Support")
st.markdown("Ask our AI agents about products, orders, or support issues.")

# Display user information
with st.expander("Your Account Information"):
    st.json(DEFAULT_USER)

# Display sample questions
with st.expander("Sample Questions You Can Ask"):
    st.markdown("""
    - What's your return policy?
    - I want to check my order status
    - Can I change my shipping address?
    - What smartphones do you sell?
    - I want to cancel my order
    - Tell me about order TG78945
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process the query asynchronously
async def process_query_async(query):
    return await process_customer_query(query, session_id=st.session_state.session_id)

# Simulated streaming function
def simulate_streaming(response, placeholder):
    # Split the response into words
    words = response.split()
    full_response = ""
    
    # Simulate streaming by adding a few words at a time
    for i in range(0, len(words), 3):
        chunk = " ".join(words[i:i+3])
        full_response += chunk + " "
        placeholder.markdown(full_response)
        time.sleep(0.05)  # Small delay to create streaming effect
    
    return full_response

# Get user input
if prompt := st.chat_input("How can we help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner while processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Processing your query..."):
            # Run the async function using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(process_query_async(prompt))
            loop.close()
        
        # Simulate streaming for better user experience
        simulate_streaming(response, message_placeholder)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add sidebar with additional information
with st.sidebar:
    st.title("About TechGadgets")
    st.markdown("""
    ## Company Information
    - **Products**: Smartphones, laptops, tablets, smart home devices, and accessories
    - **Return Policy**: 30-day return policy with receipt, items must be in original packaging
    - **Contact**: support@techgadgets.com | 1-800-TECH-HELP
    
    ## Support Hours
    Monday - Friday: 9am - 8pm EST  
    Saturday: 10am - 6pm EST  
    Sunday: Closed
    """)
    
    # Add a button to clear chat history
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = f"streamlit_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.rerun() 