import asyncio
import os
import requests
import streamlit as st
import json
from dotenv import load_dotenv

from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
from multi_agent_orchestrator.storage import InMemoryChatStorage
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = "streamlit_user_123"
if 'session_id' not in st.session_state:
    st.session_state.session_id = "streamlit_session_456"
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None

memory_storage = InMemoryChatStorage()

def create_orchestrator():
    """Creates and initializes the orchestrator with all agents with caching."""
    # Initialize the OpenAI classifier for routing requests
    custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
        api_key=os.getenv('OPENAI_API_KEY'),
    ))
    
    # Initialize the orchestrator with the classifier
    orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        MAX_RETRIES=2,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=8,
        ),
        classifier=custom_openai_classifier,
        storage=memory_storage
    )
    
    orchestrator.classifier.set_system_prompt(
        """
        You are a routing specialist for TechGadgets Inc. customer support. Your task is to analyze customer inquiries and direct them to the most appropriate agent.

        Analyze the user's input and categorize it into one of the following agent types:
        <agents>
        {{AGENT_DESCRIPTIONS}}
        </agents>

        Previous conversation context:
        <history>
        {{HISTORY}}
        </history>

        Guidelines for classification:
        1. Agent Type: Choose the most appropriate agent type based on the nature of the query.
           For follow-up responses, use the same agent type as the previous interaction.
        2. Priority: Assign based on urgency and impact.
        3. Key Entities: Extract important product names or specific issues mentioned.
        4. For follow-ups, include relevant entities from the previous interaction.
        
        Company context:
        - TechGadgets Inc. sells electronics and accessories
        - Order-related queries go to the Order Agent
        - Product info and general questions go to the Isometrik API Agent
        - Purchase-related queries go to the Sales Agent
        
        For short responses like "yes", "ok", "I want to know more", or numerical answers,
        treat them as follow-ups and maintain the previous agent selection.
        """
    )
    
    # Add agents to the orchestrator
    orchestrator.add_agent(create_isometrik_api_agent())
    
    return orchestrator

# Create Isometrik API agent
class IsometrikAPIAgent:
    def __init__(self, api_url, auth_token, agent_id):
        self.api_url = api_url
        self.auth_token = auth_token
        self.agent_id = agent_id
        self.name = 'Isometrik API Agent'
        self.description = 'External API agent for handling customer queries through Isometrik service'
        
    async def process_request(self, message: str, user_id: str, session_id: str, metadata: Dict[str, Any] = {}) -> str:
        """Process a request by sending it to the Isometrik API."""
        headers = {
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "session_id": session_id,
            "message": message,
            "agent_id": self.agent_id
        }
        
        try:
            # Run requests.post in a separate thread using asyncio.to_thread
            response = await asyncio.to_thread(
                requests.post,
                self.api_url,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                # Parse the nested JSON structure
                if isinstance(data, dict) and 'text' in data:
                    try:
                        # Parse the inner JSON string
                        inner_data = json.loads(data['text'])
                        if isinstance(inner_data, dict) and 'text' in inner_data:
                            return inner_data['text']
                        return data['text']
                    except json.JSONDecodeError:
                        return data['text']
                return str(data)
            else:
                error_text = response.text
                print(f"API Error: {response.status_code} - {error_text}")
                return f"Sorry, I encountered an error when processing your request. Please try again later."
        except Exception as e:
            print(f"Exception in API call: {str(e)}")
            return f"Sorry, I couldn't connect to our knowledge base at the moment. Please try again later."

# Wrapper to make the API agent compatible with the orchestrator
class IsometrikAPIAgentWrapper(OpenAIAgent):
    def __init__(self, api_agent: IsometrikAPIAgent, options: OpenAIAgentOptions):
        super().__init__(options)
        self.api_agent = api_agent
        self.name = api_agent.name
        self.description = api_agent.description
        
    async def process_request(self, message: str, user_id: str, session_id: str, metadata: Dict[str, Any] = {}, history: Optional[list] = None, **kwargs) -> AgentResponse:
        """Override the process_request method to use the API agent."""
        response = await self.api_agent.process_request(message, user_id, session_id, metadata)
        
        # Create a response object compatible with the orchestrator
        return response

# Function to create the Isometrik API agent
def create_isometrik_api_agent():
    # API configuration
    api_url = os.getenv('ISOMETRIK_API_URL')
    auth_token = os.getenv('ISOMETRIK_AUTH_TOKEN')
    agent_id = os.getenv('ISOMETRIK_AGENT_ID')
    
    # Create the API agent
    api_agent = IsometrikAPIAgent(api_url, auth_token, agent_id)
    
    # Create a wrapper to make it compatible with the orchestrator
    return IsometrikAPIAgentWrapper(api_agent, OpenAIAgentOptions(
        name='Isometrik API Agent',
        description='Specializes in answering customer FAQs, general inquiries, and product information using external knowledge base',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=False,  # API responses can't be streamed
        callbacks=None,  # No callbacks needed for API agent
        LOG_AGENT_DEBUG_TRACE=True,  # Enable logging for debugging
        inference_config={
            'maxTokens': 600,
            'temperature': 0.7
        }
    ))

# Streamlit UI
st.title("Isometrik Agent Chat")
st.subheader("Ask questions about TechGadgets products and services")

# Initialize orchestrator if not already done
if st.session_state.orchestrator is None:
    with st.spinner("Initializing agent..."):
        st.session_state.orchestrator = create_orchestrator()

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
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Process the request
        async def process_and_display():
            response = await st.session_state.orchestrator.route_request(
                prompt, 
                st.session_state.user_id, 
                st.session_state.session_id
            )
            
            # Extract the response text

            response_text = str(response.output)
                
            # Update the placeholder with the response
            message_placeholder.write(response_text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Run the async function
        asyncio.run(process_and_display())

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This is a demo of the Isometrik Agent integration with Streamlit.")
    st.write("The agent can answer questions about TechGadgets products and services.")
    
    st.header("Sample Questions")
    st.write("- What products do you sell?")
    st.write("- Tell me about your smartphones")
    st.write("- What's your return policy?")
    st.write("- Do you have gaming laptops?")
    
    # Add a button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()