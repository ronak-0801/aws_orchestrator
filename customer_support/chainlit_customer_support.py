import uuid
import os
import json
from datetime import datetime
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
from multi_agent_orchestrator.types import ConversationMessage

# Load environment variables
load_dotenv()

# Default user with fake order data
DEFAULT_USER = {
    "name": "Demo User",
    "email": "demo@user.com",
    "phone": "555-555-5555",
    "orders": [
        {
            "order_id": "TG78945",
            "date": "2023-07-25",
            "items": ["Gaming Laptop Pro", "Cooling Pad", "Gaming Mouse"],
            "status": "Shipped",
            "shipping_address": "789 Pine St, Testville, USA",
            "tracking_number": "FEDEX87654321"
        },
        {
            "order_id": "TG89012",
            "date": "2023-07-28",
            "items": ["Smartphone Charger", "Screen Protector"],
            "status": "Processing",
            "shipping_address": "789 Pine St, Testville, USA",
            "tracking_number": None
        }
    ]
}

# Callback handler for streaming tokens
class ChainlitAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        asyncio.run(cl.user_session.get("current_msg").stream_token(token))

# Initialize the OpenAI classifier for routing requests
custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
    api_key=os.getenv('OPENAI_API_KEY'),
    
))

# Initialize the orchestrator with the classifier
orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
    LOG_AGENT_CHAT=True,
    LOG_CLASSIFIER_CHAT=True,
    LOG_CLASSIFIER_RAW_OUTPUT=True,
    LOG_CLASSIFIER_OUTPUT=True,
    LOG_EXECUTION_TIMES=True,
    MAX_RETRIES=3,
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    MAX_MESSAGE_PAIRS_PER_AGENT=10,
    ),
    classifier=custom_openai_classifier
)

# Create query agent
def create_query_agent():
    return OpenAIAgent(OpenAIAgentOptions(
        name='Query Agent',
        description='Specializes in answering customer FAQs and general inquiries',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        inference_config={
            'maxTokens': 800,
            'temperature': 0.5
        },
        callbacks=ChainlitAgentCallbacks(),
        custom_system_prompt={
            'template': '''You are a helpful customer support agent for TechGadgets Inc. 
            Your role is to answer frequently asked questions and provide general information about our products and services.
            
            When responding to customers:
            1. Be polite, professional, and empathetic
            2. Provide clear and concise answers
            3. If you don't know the answer, say so and offer to escalate the query
            
            Company information:
            - Products: Smartphones, laptops, tablets, smart home devices, and accessories
            - Return policy: 30-day return policy with receipt, items must be in original packaging
            - Contact information: Email: support@techgadgets.com, Phone: 1-800-TECH-HELP
            
            If the customer asks about specific order details, politely explain that you'll transfer them to the Order Agent who can help with those specific inquiries.''',
        }
    ))

# Create order agent
def create_order_agent():
    # Format user data for the order agent
    user_data = json.dumps(DEFAULT_USER, indent=2)
    
    return OpenAIAgent(OpenAIAgentOptions(
        name='Order Agent',
        description='Specializes in order status, modifications, and processing',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        callbacks=ChainlitAgentCallbacks(),
        inference_config={
            'maxTokens': 800,
            'temperature': 0.4
        },
        custom_system_prompt={
            'template': '''You are an order processing specialist for TechGadgets Inc.
            Your responsibilities include:
            1. Checking order status
            2. Processing order modifications
            3. Handling cancellations and returns
            4. Addressing shipping and delivery inquiries
            
            Customer data:
            ''' + user_data + '''
            
            When handling order-related queries:
            - If the customer doesn't mention a specific order number, list their recent orders briefly
            - If they mention a specific order number, provide details for that order
            - For order modifications, explain the process clearly
            - Always be helpful and solution-oriented
            
            Shipping information:
            - Standard shipping: 3-5 business days
            - Express shipping: 1-2 business days
            - International shipping: 7-14 business days''',
        }
    ))

# Add agents to the orchestrator
orchestrator.add_agent(create_query_agent())
orchestrator.add_agent(create_order_agent())

@cl.on_chat_start
async def start():
    # Initialize session variables
    cl.user_session.set("user_id", str(uuid.uuid4()))
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("chat_history", [])
    
    # Send welcome message
    await cl.Message(
        content=f"""# Welcome to TechGadgets Customer Support
        
You're logged in as: {DEFAULT_USER['email']} (Demo User)

### Sample questions you can ask:
- What's your return policy?
- I want to check my order status
- Can I change my shipping address?
- What smartphones do you sell?
- I want to cancel my order
- Tell me about order TG78945
        """,
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Get session information
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")
    
    # Create an empty message for streaming
    msg = cl.Message(content="")
    await msg.send()  # Send the message immediately to start streaming
    cl.user_session.set("current_msg", msg)
    
    # Route the request to the appropriate agent
    response = await orchestrator.route_request(message.content, user_id, session_id, {})
    
    # Add agent name as a prefix to the message instead of metadata
    if hasattr(response, 'metadata') and hasattr(response.metadata, 'agent_name'):
        agent_name = response.metadata.agent_name
        await msg.stream_token(f"**[{agent_name}]** ")
    
    # Handle non-streaming or problematic responses
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            await msg.stream_token(response.output)
        elif isinstance(response.output, ConversationMessage):
            if hasattr(response.output, 'content') and response.output.content:
                if isinstance(response.output.content, list) and len(response.output.content) > 0:
                    if isinstance(response.output.content[0], dict) and 'text' in response.output.content[0]:
                        await msg.stream_token(response.output.content[0]['text'])
                    else:
                        await msg.stream_token(str(response.output.content[0]))
                else:
                    await msg.stream_token(str(response.output.content))
    
    # Finalize the message
    await msg.update()

if __name__ == "__main__":
    cl.run() 