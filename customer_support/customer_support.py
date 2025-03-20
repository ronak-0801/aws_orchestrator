from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions,AgentCallbacks
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import asyncio

load_dotenv()

# Initialize the OpenAI classifier for routing requests
custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
    api_key=os.getenv('OPENAI_API_KEY'),
))

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

async def process_customer_query(query, session_id=None):
    """
    Process a customer query by routing it to the appropriate agent
    
    Args:
        query (str): The customer's query
        session_id (str): Session identifier for conversation continuity
    
    Returns:
        str: The response to the customer query
    """
    # Create a session ID if none provided
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
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
        classifier=custom_openai_classifier, 

)
    
    # Create query agent
    query_agent = OpenAIAgent(OpenAIAgentOptions(
        name='Query Agent',
        description='Specializes in answering customer FAQs and general inquiries',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,  # Enable streaming
        inference_config={
            'maxTokens': 800,
            'temperature': 0.5
        },
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
    
    # Format user data for the order agent
    user_data = json.dumps(DEFAULT_USER, indent=2)
    
    # Create order agent
    order_agent = OpenAIAgent(OpenAIAgentOptions(
        name='Order Agent',
        description='Specializes in order status, modifications, and processing',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,  # Enable streaming
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
    orchestrator.add_agent(query_agent)
    orchestrator.add_agent(order_agent)
    
    print(f"\nProcessing customer query: {query}")
    
    # Route the request to the appropriate agent
    response = await orchestrator.route_request(
        query,
        user_id=DEFAULT_USER["email"],
        session_id=session_id
    )
    

    if hasattr(response, 'streaming') and response.streaming:
        print("\nSupport Agent: ", end="")
        if hasattr(response.output, 'content'):
            content = response.output.content
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and 'text' in content[0]:
                    print(content[0]['text'])
                    return content[0]['text']
                else:
                    print(content[0])
                    return content[0]
            elif isinstance(content, dict) and 'text' in content:
                print(content['text'])
                return content['text']
            else:
                print(content)
                return content
    else:
        # Extract and return the response content
        content = response.output.content
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
            return content[0]
        elif isinstance(content, dict) and 'text' in content:
            return content['text']
        
        return content

async def interactive_customer_support():
    """Run an interactive customer support session"""
    print("\n=== TechGadgets Customer Support System ===")
    print("Type 'exit' to end the conversation\n")
    
    # Generate a unique session ID for this conversation
    session_id = f"interactive_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    print(f"Logged in as: {DEFAULT_USER['email']} (Demo User)")
    print("\nSample questions you can ask:")
    print("- What's your return policy?")
    print("- I want to check my order status")
    print("- Can I change my shipping address?")
    print("- What smartphones do you sell?")
    print("- I want to cancel my order")
    print("- Tell me about order TG78945")
    
    while True:
        query = input("\nHow can we help you today? ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for contacting TechGadgets support. Have a great day!")
            break
            
        response = await process_customer_query(query, session_id=session_id)
        # The response is already printed in the process_customer_query function

async def main():
    await interactive_customer_support()

if __name__ == "__main__":
    asyncio.run(main())
