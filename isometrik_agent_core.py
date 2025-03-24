import asyncio
import os
import requests
import json
from dotenv import load_dotenv

from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
from multi_agent_orchestrator.storage import InMemoryChatStorage
from typing import Dict, Any, Optional, List

# Load environment variables
load_dotenv()

# Initialize storage
memory_storage = InMemoryChatStorage()

def create_formatter_agent():
    """Creates a formatter agent to enhance responses for better readability."""
    return OpenAIAgent(OpenAIAgentOptions(
        name='Response Formatter Agent',
        description='Specializes in formatting responses to be more readable and engaging using proper markdown',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=False,
        LOG_AGENT_DEBUG_TRACE=True,
        custom_system_prompt={
            'template': '''You are a response formatter for customer service interactions. Your job is to:
            
            1. Format raw responses to be more readable and engaging
            2. Organize information with proper markdown formatting (headers, bullet points, etc.)
            3. Highlight key information like prices, product features, or policy details
            4. Ensure any links are properly formatted and accessible
            5. Maintain all factual information from the original response
            6. Keep the tone friendly and helpful
            
            Do not add any fictional information not present in the original response.
            
            When formatting:
            - Use markdown headers (# and ##) for main sections
            - Use bullet points for lists of features or options
            - Bold important information like prices or key details
            - Create tables when comparing multiple products or options
            - Format code snippets or technical information in code blocks
            - Ensure proper spacing for readability
            '''
        }
    ))

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
        - Responses that need formatting or better presentation go to the Response Formatter Agent
        
        Special routing rules:
        1. If a response from the Isometrik API Agent needs formatting, route it to the Response Formatter Agent
        2. For raw data or complex information, always use the Response Formatter Agent after getting the initial response
        
        For short responses like "yes", "ok", "I want to know more", or numerical answers,
        treat them as follow-ups and maintain the previous agent selection.
        """
    )
    
    # Add agents to the orchestrator
    orchestrator.add_agent(create_isometrik_api_agent())
    orchestrator.add_agent(create_formatter_agent())
    
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
                
                inner_data = json.loads(data['text'])
                return inner_data
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
        return AgentResponse(
            output=response,
            streaming=False,
            metadata=metadata
        )

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
    ))

class IsometrikChatSession:
    def __init__(self):
        self.orchestrator = create_orchestrator()
        self.messages = []
        self.user_id = "default_user_123"
        self.session_id = "default_session_456"
    
    def set_user_id(self, user_id: str):
        """Set the user ID for this chat session."""
        self.user_id = user_id
    
    def set_session_id(self, session_id: str):
        """Set the session ID for this chat session."""
        self.session_id = session_id
    
    async def process_message(self, message: str) -> str:
        """Process a user message and return the agent's response."""
        try:
            # Add user message to chat history
            self.messages.append({"role": "user", "content": message})
            
            # Process the request
            response = await self.orchestrator.route_request(
                message, 
                self.user_id, 
                self.session_id
            )
            
            # Extract the response text
            if isinstance(response, AgentResponse):
                response_text = response.output
            else:
                response_text = str(response)
            
            # Add assistant response to chat history
            self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            self.messages.append({"role": "assistant", "content": error_message})
            return error_message
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history for this session."""
        return self.messages
    
    def clear_chat_history(self):
        """Clear the chat history for this session."""
        self.messages = []


# Example usage as a CLI application
async def main():
    print("Welcome to TechGadgets Customer Support!")
    print("Type 'exit' to quit, 'clear' to clear chat history.")
    
    chat_session = IsometrikChatSession()
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Thank you for using TechGadgets Customer Support. Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            chat_session.clear_chat_history()
            print("Chat history cleared.")
            continue
        
        print("\nProcessing your request...")
        response = await chat_session.process_message(user_input)
        print(f"\nAssistant: {response}")


# Example usage as an API
class IsometrikAgentAPI:
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id: str, user_id: str = None) -> IsometrikChatSession:
        """Get an existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = IsometrikChatSession()
            self.sessions[session_id].set_session_id(session_id)
            if user_id:
                self.sessions[session_id].set_user_id(user_id)
        return self.sessions[session_id]
    
    async def process_message(self, session_id: str, message: str, user_id: str = None) -> str:
        """Process a message for a specific session."""
        session = self.get_or_create_session(session_id, user_id)
        return await session.process_message(message)
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the chat history for a specific session."""
        if session_id in self.sessions:
            return self.sessions[session_id].get_chat_history()
        return []
    
    def clear_chat_history(self, session_id: str):
        """Clear the chat history for a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear_chat_history()


# Custom Retriever using Isometrik API
class IsometrikRetriever:
    def __init__(self, api_url, auth_token, agent_id):
        self.api_url = api_url
        self.auth_token = auth_token
        self.agent_id = agent_id
        
    async def retrieve(self, query: str, user_id: str = "default_user", session_id: str = "default_session") -> List[Dict[str, Any]]:
        """Retrieve information from Isometrik API based on the query."""
        headers = {
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "session_id": session_id,
            "message": query,
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
                inner_data = json.loads(data['text'])
                
                # Format the response as a document for the retriever
                return [{
                    "content": inner_data,
                    "metadata": {
                        "source": "isometrik_api",
                        "query": query
                    }
                }]
            else:
                error_text = response.text
                print(f"API Error: {response.status_code} - {error_text}")
                return [{
                    "content": f"Error retrieving information: {error_text}",
                    "metadata": {
                        "source": "isometrik_api",
                        "error": True
                    }
                }]
        except Exception as e:
            print(f"Exception in API call: {str(e)}")
            return [{
                "content": f"Error connecting to knowledge base: {str(e)}",
                "metadata": {
                    "source": "isometrik_api",
                    "error": True
                }
            }]

# Create a main agent with Isometrik retriever
def create_main_agent_with_retriever():
    """Creates a main agent that uses Isometrik as a retriever."""
    # Create the Isometrik retriever
    retriever = IsometrikRetriever(
        api_url=os.getenv('ISOMETRIK_API_URL'),
        auth_token=os.getenv('ISOMETRIK_AUTH_TOKEN'),
        agent_id=os.getenv('ISOMETRIK_AGENT_ID')
    )
    
    # Create the main agent with the retriever
    return OpenAIAgent(OpenAIAgentOptions(
        name='TechGadgets Assistant',
        description='A customer support assistant with access to TechGadgets knowledge base',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        streaming=True,
        retriever=retriever,
        LOG_AGENT_DEBUG_TRACE=True,
        custom_system_prompt={
            'template': '''You are a customer support specialist for TechGadgets Inc.
            
            When answering customer questions, you will use the knowledge base to retrieve accurate information.
            
            Guidelines for using retrieved information:
            1. Use the retrieved content to provide accurate answers about products, policies, and services
            2. Format your responses in a clear, readable way using markdown
            3. Highlight important information like prices, features, or policy details
            4. If the retrieved information contains errors or is incomplete, acknowledge this and provide the best answer you can
            5. If no relevant information is retrieved, be honest about not having the specific information
            
            Your tone should be:
            - Professional but friendly
            - Helpful and solution-oriented
            - Clear and concise
            
            When formatting responses:
            - Use headers for main sections
            - Use bullet points for lists
            - Bold important information
            - Create tables for comparing options when appropriate
            '''
        }
    ))

# Create a chat session using the main agent with retriever
class MainAgentChatSession:
    def __init__(self):
        self.agent = create_main_agent_with_retriever()
        self.messages = []
        self.user_id = "default_user_123"
        self.session_id = "default_session_456"
    
    def set_user_id(self, user_id: str):
        """Set the user ID for this chat session."""
        self.user_id = user_id
    
    def set_session_id(self, session_id: str):
        """Set the session ID for this chat session."""
        self.session_id = session_id
    
    async def process_message(self, message: str) -> str:
        """Process a user message and return the agent's response."""
        try:
            # Add user message to chat history
            self.messages.append({"role": "user", "content": message})
            
            # Process the request with the main agent
            response = await self.agent.process_request(
                message, 
                self.user_id, 
                self.session_id,
                metadata={"retriever_context": {"user_id": self.user_id, "session_id": self.session_id}}
            )
            
            # Extract the response text
            if isinstance(response, AgentResponse):
                response_text = response.output
            else:
                response_text = str(response)
            
            # Add assistant response to chat history
            self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            self.messages.append({"role": "assistant", "content": error_message})
            return error_message
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history for this session."""
        return self.messages
    
    def clear_chat_history(self):
        """Clear the chat history for this session."""
        self.messages = []

# Example usage of the main agent with retriever as a CLI application
async def main_with_retriever():
    print("Welcome to TechGadgets Customer Support!")
    print("Type 'exit' to quit, 'clear' to clear chat history.")
    
    chat_session = MainAgentChatSession()
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Thank you for using TechGadgets Customer Support. Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            chat_session.clear_chat_history()
            print("Chat history cleared.")
            continue
        
        print("\nProcessing your request...")
        response = await chat_session.process_message(user_input)
        print(f"\nAssistant: {response}")

# Example usage as an API with the main agent
class MainAgentAPI:
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id: str, user_id: str = None) -> MainAgentChatSession:
        """Get an existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = MainAgentChatSession()
            self.sessions[session_id].set_session_id(session_id)
            if user_id:
                self.sessions[session_id].set_user_id(user_id)
        return self.sessions[session_id]
    
    async def process_message(self, session_id: str, message: str, user_id: str = None) -> str:
        """Process a message for a specific session."""
        session = self.get_or_create_session(session_id, user_id)
        return await session.process_message(message)
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the chat history for a specific session."""
        if session_id in self.sessions:
            return self.sessions[session_id].get_chat_history()
        return []
    
    def clear_chat_history(self, session_id: str):
        """Clear the chat history for a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear_chat_history()

# Run the main agent with retriever if this file is executed directly
if __name__ == "__main__":
    # Choose which implementation to run
    use_retriever_approach = True
    
    if use_retriever_approach:
        asyncio.run(main_with_retriever())
    else:
        asyncio.run(main()) 