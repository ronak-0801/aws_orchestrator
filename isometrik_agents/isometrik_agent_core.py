import asyncio
import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse, Agent, AgentOptions
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
from multi_agent_orchestrator.storage import InMemoryChatStorage
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.retrievers import Retriever

# Load environment variables
load_dotenv()

# Initialize storage
memory_storage = InMemoryChatStorage()

# Custom options class for Isometrik Agent

class IsometrikRetrieverOptions:
    def __init__(self, endpoint: str, auth_token: str, agent_id: str):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.agent_id = agent_id

class IsometrikRetriever(Retriever):
    def __init__(self, options: IsometrikRetrieverOptions):
        super().__init__(options)
        self.options = options
        
        if not all([options.endpoint, options.auth_token, options.agent_id]):
            raise ValueError("endpoint, auth_token, and agent_id are required in options")
        
    async def retrieve(self, text: str) -> List[Dict[str, Any]]:
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.options.auth_token}'
            }
            
            payload = {
                "session_id": "retrieval_session",  # Using the same payload structure as chat
                "message": text,
                "agent_id": self.options.agent_id
            }
            
            print(f"Retriever: Sending request to {self.options.endpoint} for agent {self.options.agent_id}")
            response = await asyncio.to_thread(
                requests.post,
                self.options.endpoint,  
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Retriever error: HTTP {response.status_code} - {response.text}")
                raise Exception(f"HTTP error! status: {response.status_code}")
            
            # Parse the response according to your API's format
            content = response.text
            try:
                data = json.loads(content)
                if 'text' in data:
                    inner_data = json.loads(data['text'])
                    return [{"content": inner_data}]
                return [{"content": data}]
            except json.JSONDecodeError:
                return [{"content": content}]
            
        except Exception as e:
            print(f"Retriever error: {str(e)}")
            raise Exception(f"Failed to retrieve: {str(e)}")

    async def retrieve_and_combine_results(self, text: str) -> str:
        try:
            results = await self.retrieve(text)
            combined = "\n".join(str(result.get('content', '')) for result in results)
            print(f"Retriever: Combined {len(results)} results")
            return combined
        except Exception as e:
            print(f"Retriever combine error: {str(e)}")
            raise Exception(f"Failed to retrieve and combine results: {str(e)}")

    async def retrieve_and_generate(self, text: str) -> str:
        try:
            return await self.retrieve_and_combine_results(text)
        except Exception as e:
            print(f"Retriever generate error: {str(e)}")
            raise Exception(f"Failed to retrieve and generate: {str(e)}")

# Create retriever instances for each agent
def create_query_retriever():
    return IsometrikRetriever(IsometrikRetrieverOptions(
        endpoint=os.getenv('QUERY_AGENT_API_URL'),
        auth_token=os.getenv('QUERY_AGENT_AUTH_TOKEN'),
        agent_id=os.getenv('QUERY_AGENT_ID')
    ))

def create_order_retriever():
    return IsometrikRetriever(IsometrikRetrieverOptions(
        endpoint=os.getenv('ORDER_AGENT_API_URL'),
        auth_token=os.getenv('ORDER_AGENT_AUTH_TOKEN'),
        agent_id=os.getenv('ORDER_AGENT_ID')
    ))

def create_ecom_manager_retriever():
    return IsometrikRetriever(IsometrikRetrieverOptions(
        endpoint=os.getenv('MANAGER_AGENT_API_URL'),
        auth_token=os.getenv('MANAGER_AGENT_AUTH_TOKEN'),
        agent_id=os.getenv('MANAGER_AGENT_ID')
    ))

def create_subscription_retriever():
    return IsometrikRetriever(IsometrikRetrieverOptions(
        endpoint=os.getenv('SUBSCRIPTION_AGENT_API_URL'),
        auth_token=os.getenv('SUBSCRIPTION_AGENT_AUTH_TOKEN'),
        agent_id=os.getenv('SUBSCRIPTION_AGENT_ID')
    ))
    
def create_product_retriever():
    return IsometrikRetriever(IsometrikRetrieverOptions(
        endpoint=os.getenv('PRODUCT_AGENT_API_URL'),
        auth_token=os.getenv('PRODUCT_AGENT_AUTH_TOKEN'),
        agent_id=os.getenv('PRODUCT_AGENT_ID')
    ))

# Update the agent creation functions to use retrievers
def create_query_agent(streaming_handler=None):
    return OpenAIAgent(OpenAIAgentOptions(
        name='Query Agent',
        description='Specializes in answering customer FAQs, general inquiries, and product information',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True if streaming_handler else False,
        callbacks=streaming_handler,
        retriever=create_query_retriever()
    ))

def create_order_agent(streaming_handler=None):
    return OpenAIAgent(OpenAIAgentOptions(
        name='Order Agent',
        description='Specializes in handling order-related queries, order status, and processing new orders',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True if streaming_handler else False,
        callbacks=streaming_handler,
        retriever=create_order_retriever()
    ))

def create_manager_agent(streaming_handler=None):
    return OpenAIAgent(OpenAIAgentOptions(
        name='Manager Agent',
        description='Specializes in finding toxin-free, eco-friendly solutions for home and personal care',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True if streaming_handler else False,
        callbacks=streaming_handler,
        retriever=create_ecom_manager_retriever()
    ))

def create_subscription_agent(streaming_handler=None):
    return OpenAIAgent(OpenAIAgentOptions(
        name='Subscription Agent',
        description='Specializes in handling subscription-related queries',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True if streaming_handler else False,
        callbacks=streaming_handler,
        retriever=create_subscription_retriever()
    ))

def create_product_agent(streaming_handler=None):
    return OpenAIAgent(OpenAIAgentOptions(
        name='Product Agent',
        description='Specializes in providing detailed product information including ingredients, usage instructions, benefits, and specifications',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True if streaming_handler else False,
        callbacks=streaming_handler,    
        retriever=create_product_retriever()
    ))

def create_orchestrator(streaming_handler=None):
    """Creates and initializes the orchestrator with all agents with streaming support."""
    # Initialize the OpenAI classifier for routing requests
    custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_id="gpt-4o-mini",
    ))
    
    # Initialize the orchestrator with the classifier
    orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        MAX_RETRIES=2,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=8,
        ),
        classifier=custom_openai_classifier,
        storage=memory_storage
    )
    
    # Add all agents with streaming support
    query_agent = create_query_agent(streaming_handler)
    order_agent = create_order_agent(streaming_handler)
    manager_agent = create_manager_agent(streaming_handler)
    subscription_agent = create_subscription_agent(streaming_handler)
    product_agent = create_product_agent(streaming_handler)
    
    orchestrator.add_agent(query_agent)
    orchestrator.add_agent(order_agent)
    orchestrator.add_agent(manager_agent)
    orchestrator.add_agent(subscription_agent)
    orchestrator.add_agent(product_agent)
    
    # Update the classifier prompt following Anthropic's best practices
    orchestrator.classifier.set_system_prompt(
        """
        You are AgentMatcher, an intelligent assistant designed to analyze user queries and match them with
        the most suitable agent or department. Your task is to understand the user's request,
        identify key entities and intents, and determine which agent would be best equipped
        to handle the query.

        Important: The user's input may be a follow-up response to a previous interaction.
        The conversation history, including the name of the previously selected agent, is provided.
        If the user's input appears to be a continuation of the previous conversation
        (e.g., "yes", "ok", "I want to know more", "1"), select the same agent as before.

        Analyze the user's input and categorize it into one of the following agent types:
        <agents>
        {{AGENT_DESCRIPTIONS}}
        </agents>

        Guidelines for classification:

        1. Agent Type: Choose the most appropriate agent based on the nature of the query.
           - Query Agent: General product information, FAQs, and basic inquiries
           - Order Agent: Order status, tracking, processing, and shipping
           - Manager Agent: Eco-friendly products, toxin-free solutions, sustainable living
           - Subscription Agent: Subscription-related queries
           - Product Agent: Detailed product information including ingredients, usage instructions, benefits, and specifications

        2. Priority: Assign based on urgency and impact.
           - High: Issues affecting service, urgent requests
           - Medium: Non-urgent product inquiries, general questions
           - Low: Information requests, browsing

        3. Key Entities: Extract important product names, issues, or specific requests mentioned.
           For follow-ups, include relevant entities from previous interactions.

        4. Confidence: Indicate how confident you are in the classification.
           - High: Clear, straightforward requests or clear follow-ups
           - Medium: Requests with some ambiguity but likely classification
           - Low: Vague or multi-faceted requests that could fit multiple categories

        5. Is Followup: Indicate whether the input is a follow-up to a previous interaction.

        Handle variations in user input, including different phrasings, synonyms,
        and potential spelling errors.

        For short responses like "yes", "ok", "I want to know more", or numerical answers,
        treat them as follow-ups and maintain the previous agent selection.

        Here is the conversation history that you need to take into account before answering:
        <history>
        {{HISTORY}}
        </history>

        Skip any preamble and provide only the response in the specified format.
        """
    )
    
    return orchestrator

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
            
            # Process the request with the Isometrik API Agent first
            print(f"Routing request to orchestrator: {message}")
            response = await self.orchestrator.route_request(
                message, 
                self.user_id, 
                self.session_id
            )
            
            # Extract the response text
            if isinstance(response, AgentResponse):
                if isinstance(response.output, ConversationMessage):
                    # Extract text from the ConversationMessage
                    if response.output.content and len(response.output.content) > 0:
                        response_text = response.output.content[0].get("text", "")
                    else:
                        response_text = "No content in response"
                else:
                    # Fallback for other response types
                    response_text = str(response.output)
            else:
                response_text = str(response)
                        
            # Add assistant response to chat history
            self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
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

# Run the CLI example if this file is executed directly
if __name__ == "__main__":
    asyncio.run(main()) 