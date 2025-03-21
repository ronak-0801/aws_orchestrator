import uuid
import os
import json
import asyncio
import requests
import threading
import time
import chainlit as cl
from dotenv import load_dotenv
from datetime import datetime, timedelta
from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions, AgentCallbacks, AgentResponse
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.storage import InMemoryChatStorage

# Load environment variables
load_dotenv()

# Use a single memory storage instance for the whole application
memory_storage = InMemoryChatStorage()

# Default user info
DEFAULT_USER = {
    "name": "Demo User",
    "email": "demo@user.com",
    "phone": "555-555-5555"
}

# API endpoints
PRODUCTS_API = "https://dummyjson.com/products?limit=100"  # Get up to 100 products

# In-memory storage for user purchases with product cache
user_purchases = {}
_products_cache = None
_products_cache_time = 0
_products_cache_ttl = 300  # Cache TTL in seconds (5 minutes)

# Order status sequence with shortened durations for demo
ORDER_STATUS_SEQUENCE = [
    {"status": "Processing", "duration": 5},
    {"status": "Preparing for Shipment", "duration": 5},
    {"status": "Shipped", "duration": 5},
    {"status": "Out for Delivery", "duration": 5},
    {"status": "Delivered", "duration": 0}
]

# Function to update order status over time
def update_order_status(user_id, order_id):
    """Updates the status of an order over time."""
    if user_id not in user_purchases:
        return
    
    # Find the order
    order = None
    for purchase in user_purchases[user_id]:
        if purchase["order_id"] == order_id:
            order = purchase
            break
    
    if not order:
        return
    
    # Update the status through the sequence
    for i, status_info in enumerate(ORDER_STATUS_SEQUENCE):
        # Skip the first status (Processing) as it's already set
        if i > 0:
            # Update the status
            order["status"] = status_info["status"]
            print(f"Order {order_id} status updated to: {status_info['status']}")
        
        # If this is the final status, stop
        if status_info["duration"] == 0:
            break
        
        # Wait for the specified duration
        time.sleep(status_info["duration"])

# Function to fetch products with caching
def fetch_products():
    """Fetches product information from the DummyJSON API with caching."""
    global _products_cache, _products_cache_time
    
    # Return cached products if available and not expired
    current_time = time.time()
    if _products_cache and (current_time - _products_cache_time) < _products_cache_ttl:
        return _products_cache
    
    try:
        response = requests.get(PRODUCTS_API)
        if response.status_code == 200:
            data = response.json()
            products = data.get('products', [])
            # Create a dictionary with product ID as key for easy lookup
            product_dict = {product['id']: product for product in products}
            
            # Update cache
            _products_cache = product_dict
            _products_cache_time = current_time
            
            return product_dict
        else:
            print(f"Failed to fetch products. Status code: {response.status_code}")
            return _products_cache or {}
    except Exception as e:
        print(f"Error fetching products: {str(e)}")
        return _products_cache or {}

# Function to get user purchases
def get_user_purchases(user_id):
    """Gets the purchase history for a user."""
    return user_purchases.get(user_id, [])

# Function to add a purchase for a user
def add_user_purchase(user_id, product_id, product_name, quantity, price):
    """Adds a purchase to a user's purchase history."""
    if user_id not in user_purchases:
        user_purchases[user_id] = []
    
    # Generate a unique order ID
    order_id = f"TG{str(uuid.uuid4())[:6].upper()}"
    
    # Calculate estimated delivery date (5 days from now)
    delivery_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
    
    purchase = {
        "order_id": order_id,
        "product_id": product_id,
        "product_name": product_name,
        "quantity": quantity,
        "price": price,
        "total": price * quantity,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "estimated_delivery": delivery_date,
        "status": "Processing"
    }
    
    user_purchases[user_id].append(purchase)
    
    # Start a thread to update the order status over time
    status_thread = threading.Thread(
        target=update_order_status,
        args=(user_id, order_id),
        daemon=True
    )
    status_thread.start()
    
    return purchase

# Optimized callback handler for streaming tokens
class ChainlitAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # Reduced delay for faster response time
        time.sleep(0.005)  # 5ms delay between tokens - can be adjusted based on user preferences
        asyncio.run(cl.user_session.get("current_msg").stream_token(token))

# Orchestrator and agent creation logic with caching
_agent_cache = {}
_agent_cache_time = {}
_agent_cache_ttl = 60  # Cache TTL in seconds (1 minute)

def create_orchestrator(user_id, force_refresh=False):
    """Creates and initializes the orchestrator with all agents with caching."""
    global _agent_cache, _agent_cache_time
    
    current_time = time.time()
    cache_key = f"orchestrator_{user_id}"
    
    # Return cached orchestrator if available, not expired, and not forced to refresh
    if not force_refresh and cache_key in _agent_cache and (current_time - _agent_cache_time.get(cache_key, 0)) < _agent_cache_ttl:
        return _agent_cache[cache_key]
    
    # Initialize the OpenAI classifier for routing requests
    custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
        api_key=os.getenv('OPENAI_API_KEY'),
    ))
    
    # Initialize the orchestrator with the classifier
    orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        MAX_RETRIES=2,  # Reduced from 3 to 2
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=8,  # Reduced from 10 to 8
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
        - Product info and general questions go to the Query Agent
        - Purchase-related queries go to the Sales Agent
        
        For short responses like "yes", "ok", "I want to know more", or numerical answers,
        treat them as follow-ups and maintain the previous agent selection.
        """
    )
    
    # Add agents to the orchestrator
    orchestrator.add_agent(create_query_agent())
    orchestrator.add_agent(create_order_agent(user_id))
    orchestrator.add_agent(create_sales_agent(user_id))
    
    # Update cache
    _agent_cache[cache_key] = orchestrator
    _agent_cache_time[cache_key] = current_time
    
    return orchestrator

# Create query agent with optimized product info handling
def create_query_agent():
    # Fetch products for the query agent
    products = fetch_products()
    
    # More efficient category and popular product extraction
    product_categories = set()
    popular_products = []
    
    # Get a list of products
    product_list = list(products.values())
    
    # Extract categories and select popular products (limit to what's needed)
    for product in product_list[:20]:  # Limited to first 20 
        if 'category' in product:
            product_categories.add(product['category'])
        
        if len(popular_products) < 5 and 'title' in product and product.get('rating', 0) >= 4.5:
            popular_products.append({
                'name': product['title'],
                'price': product.get('price', 'N/A'),
                'category': product.get('category', 'Unknown'),
                'rating': product.get('rating', 0)
            })
    
    # Build product info once
    product_info = {
        'categories': list(product_categories),
        'popular_products': popular_products
    }
    
    product_info_json = json.dumps(product_info, indent=2)
    
    return OpenAIAgent(OpenAIAgentOptions(
        name='Query Agent',
        description='Specializes in answering customer FAQs, general inquiries, and product information',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',  # Consider using a faster model if available
        streaming=True,
        callbacks=ChainlitAgentCallbacks(),
        LOG_AGENT_DEBUG_TRACE=False,  # Reduced logging to improve performance
        inference_config={
            'maxTokens': 600,  # Reduced from 800
            'temperature': 0.7
        },
        custom_system_prompt={
            'template': '''You are a customer support specialist for TechGadgets Inc.
            Your responsibilities include:
            1. Answering general questions about the company
            2. Providing information about products and services
            3. Explaining policies and procedures
            4. Addressing basic technical questions
            
            Company information:
            - TechGadgets Inc. is an electronics retailer founded in 2010
            - We sell smartphones, laptops, tablets, gaming consoles, and accessories
            - We have physical stores in major cities and an online store
            - Our headquarters is in San Francisco, CA
            
            Product information:
            ''' + product_info_json + '''
            
            Policies:
            - Return policy: 30-day return window for most products
            - Warranty: Manufacturer warranty applies to all products
            - Shipping: Free shipping on orders over $50
            - Price match: We match prices from major competitors
            
            When answering:
            - Be concise and specific
            - For detailed product specifications, recommend checking our website''',
        }
    ))

# Create order agent with optimized order formatting
def create_order_agent(user_id):
    # Get user purchases - this should have the latest status
    purchases = get_user_purchases(user_id)
    
    # Format orders for the agent - only include necessary fields
    formatted_orders = []
    
    # Add user purchases as orders (only the most recent 5 to reduce token usage)
    for purchase in purchases[-5:]:
        formatted_order = {
            "order_id": purchase["order_id"],
            "date": purchase["date"],
            "status": purchase["status"],
            "total_price": purchase["total"],
            "estimated_delivery": purchase.get("estimated_delivery", ""),
            "items": [{
                "product_name": purchase["product_name"],
                "quantity": purchase["quantity"],
                "price": purchase["price"]
            }]
        }
        formatted_orders.append(formatted_order)
    
    # Sort orders by date (newest first)
    formatted_orders.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    # Combine user data with orders
    user_data = {
        **DEFAULT_USER,
        "orders": formatted_orders
    }
    
    # Format user data for the order agent
    formatted_user_data = json.dumps(user_data, indent=2)
    
    return OpenAIAgent(OpenAIAgentOptions(
        name='Order Agent',
        description='Specializes in order status, modifications, and processing',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        callbacks=ChainlitAgentCallbacks(),
        LOG_AGENT_DEBUG_TRACE=False,  # Reduced logging
        inference_config={
            'maxTokens': 600,  # Reduced from 800
            'temperature': 0.4
        },
        custom_system_prompt={
            'template': '''You are an order processing specialist for TechGadgets Inc.
            Your responsibilities include:
            1. Checking order status
            2. Processing order modifications
            3. Handling cancellations and returns
            4. Addressing shipping and delivery inquiries
            
            Customer data including orders:
            ''' + formatted_user_data + '''
            
            When handling order-related queries:
            - Be concise but thorough
            - For order status, show the order ID, status, and estimated delivery
            - For specific order numbers, provide complete details
            - Always be solution-oriented
            
            Order status progression:
            Processing → Preparing for Shipment → Shipped → Out for Delivery → Delivered''',
        }
    ))

# Create sales agent with optimized product handling
def create_sales_agent(user_id):
    # Fetch products for the sales agent
    products = fetch_products()
    
    # Get user purchases (limit to recent 3 to reduce data size)
    purchases = get_user_purchases(user_id)[-3:]
    
    # Format product data for the sales agent - select key fields only
    formatted_products = []
    for product_id, product in list(products.items())[:20]:  # Limit to 20 products
        formatted_product = {
            'id': product_id,
            'name': product.get('title', f'Product {product_id}'),
            'price': product.get('price', 0),
            'category': product.get('category', 'Unknown'),
            'description': product.get('description', '')[:80] + '...' if len(product.get('description', '')) > 80 else product.get('description', ''),
            'rating': product.get('rating', 0),
            'brand': product.get('brand', 'Unknown')
        }
        formatted_products.append(formatted_product)
    
    products_json = json.dumps(formatted_products, indent=2)
    purchases_json = json.dumps(purchases, indent=2)
    
    return OpenAIAgent(OpenAIAgentOptions(
        name='Sales Agent',
        description='Specializes in product recommendations and purchase assistance',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        callbacks=ChainlitAgentCallbacks(),
        LOG_AGENT_DEBUG_TRACE=False,  # Reduced logging
        inference_config={
            'maxTokens': 600,  # Reduced from 800
            'temperature': 0.5
        },
        custom_system_prompt={
            'template': '''You are a sales specialist for TechGadgets Inc.
            Your responsibilities include:
            1. Recommending products based on customer needs
            2. Providing product comparisons
            3. Explaining product features and benefits
            4. Assisting with the purchase process
            
            Available products:
            ''' + products_json + '''
            
            Customer's previous purchases:
            ''' + purchases_json + '''
            
            For purchase requests:
            - When a customer wants to buy, ask for quantity if not specified
            - If quantity not specified, assume 1 unit
            - Ask for purchase confirmation with a simple yes/no question
            - Once confirmed, process the order
            - Provide a purchase confirmation with details
            - Let them know they can check order status with the Order Agent
            
            This is a demo system - keep the purchase process simple.''',
        }
    ))

@cl.on_chat_start
async def start():
    # Initialize session variables
    user_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("pending_purchase", None)
    
    # Pre-fetch products to warm up the cache
    fetch_products()
    
    # Create orchestrator upfront to avoid delays in first interaction
    orchestrator = create_orchestrator(user_id)
    cl.user_session.set("orchestrator", orchestrator)
    
    # Send welcome message
    await cl.Message(
        content="""# Welcome to TechGadgets Customer Support
        
You're logged in as: demo@user.com (Demo User)

### Sample questions you can ask:
- What's your return policy?
- Check my order status
- Can I change my shipping address?
- What products do you recommend for gaming?
- I want to cancel my order
- Show me my recent orders
- I'm looking to buy a new smartphone
- What's the status of my latest order?
        """,
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Get session information
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")
    
    # Get the existing orchestrator (or create if needed)
    orchestrator = cl.user_session.get("orchestrator")
    if not orchestrator:
        orchestrator = create_orchestrator(user_id)
        cl.user_session.set("orchestrator", orchestrator)
    
    # Create an empty message for streaming
    msg = cl.Message(content="")
    await msg.send()
    cl.user_session.set("current_msg", msg)
    
    # Check for pending purchase intent
    pending_purchase = cl.user_session.get("pending_purchase", None)
    
    # Handle purchase confirmation
    if pending_purchase and any(word in message.content.lower() for word in ["yes", "confirm", "sure", "ok", "okay", "proceed", "buy it"]):
        # Add the purchase to the user's history
        product_id = pending_purchase["product_id"]
        product_name = pending_purchase["product_name"]
        quantity = pending_purchase["quantity"]
        price = pending_purchase["price"]
        
        # Create the purchase
        purchase = add_user_purchase(user_id, product_id, product_name, quantity, price)
        
        # Clear the pending purchase
        cl.user_session.set("pending_purchase", None)
        
        # Force a refresh of the orchestrator to include the new purchase
        orchestrator = create_orchestrator(user_id, force_refresh=True)
        cl.user_session.set("orchestrator", orchestrator)
        
        # Send purchase confirmation
        confirmation = f"""# Purchase Confirmed!

Your order has been processed.

**Order Details:**
- **Order ID:** {purchase['order_id']}
- **Product:** {product_name}
- **Quantity:** {quantity}
- **Price:** ${price}
- **Total:** ${price * quantity}
- **Estimated Delivery:** {purchase['estimated_delivery']}

Thank you for your purchase! You can check your order status anytime.
"""
        # Stream the confirmation with a faster pace
        for line in confirmation.split('\n'):
            await msg.stream_token(line + '\n')
            await asyncio.sleep(0.05)  # 50ms delay between lines - faster than before
        
        await msg.update()
        return
    
    # Check if the message is asking about orders
    if any(phrase in message.content.lower() for phrase in ["my orders", "my order", "order status", "check order", "show orders", "view orders"]):
        # Force refresh the orchestrator to get the latest order data
        orchestrator = create_orchestrator(user_id, force_refresh=True)
        cl.user_session.set("orchestrator", orchestrator)
    
    # Route the request to the appropriate agent
    response = await orchestrator.route_request(message.content, user_id, session_id, {})
    
    # Check if the message contains purchase intent - more specific patterns
    if "buy" in message.content.lower() or "purchase" in message.content.lower() or "get me" in message.content.lower():
        # Extract product information from the message
        products = fetch_products()
        for product_id, product in products.items():
            product_name = product.get('title', '').lower()
            if product_name in message.content.lower():
                # Check if quantity is specified
                quantity = 1  # Default quantity
                
                # Simplified quantity extraction
                import re
                quantity_patterns = [
                    r'(\d+)\s+' + re.escape(product_name),
                    r'buy\s+(\d+)',
                    r'get\s+(\d+)',
                    r'order\s+(\d+)',
                    r'purchase\s+(\d+)'
                ]
                
                for pattern in quantity_patterns:
                    quantity_match = re.search(pattern, message.content.lower())
                    if quantity_match:
                        try:
                            quantity = int(quantity_match.group(1))
                            break
                        except:
                            pass
                
                # Store the purchase intent for confirmation
                cl.user_session.set("pending_purchase", {
                    "product_id": product_id,
                    "product_name": product.get('title'),
                    "quantity": quantity,
                    "price": product.get('price', 0)
                })
                
                # Add a confirmation request to the response
                confirmation_request = f"\n\nI see you want to purchase {quantity} {product.get('title')} at ${product.get('price', 0)} each. Would you like to confirm this purchase? (Say 'yes' to confirm)"
                await msg.stream_token(confirmation_request)
                break
    
    # Handle non-streaming responses
    if isinstance(response, AgentResponse) and response.streaming is False:
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