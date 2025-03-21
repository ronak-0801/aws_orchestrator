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

memory_storage = InMemoryChatStorage()

# Default user info
DEFAULT_USER = {
    "name": "Demo User",
    "email": "demo@user.com",
    "phone": "555-555-5555"
}

# API endpoints
PRODUCTS_API = "https://dummyjson.com/products?limit=100"  # Get up to 100 products

# In-memory storage for user purchases
user_purchases = {}

# Order status progression with much shorter durations for testing
ORDER_STATUS_SEQUENCE = [
    {"status": "Processing", "duration": 5},  # 5 seconds
    {"status": "Preparing for Shipment", "duration": 5},  # 5 seconds
    {"status": "Shipped", "duration": 5},  # 5 seconds
    {"status": "Out for Delivery", "duration": 5},  # 5 seconds
    {"status": "Delivered", "duration": 0}  # Final state
]

# Global storage for status updates
status_updates = {}

# Function to update order status over time
def update_order_status(user_id, order_id):
    """
    Updates the status of an order over time.
    
    :param user_id: The user ID
    :param order_id: The order ID
    """
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

# Function to fetch products from the DummyJSON API
def fetch_products():
    """
    Fetches product information from the DummyJSON API.
    
    :return: A dictionary mapping product IDs to product details
    """
    try:
        response = requests.get(PRODUCTS_API)
        if response.status_code == 200:
            data = response.json()
            products = data.get('products', [])
            # Create a dictionary with product ID as key for easy lookup
            product_dict = {product['id']: product for product in products}
            return product_dict
        else:
            print(f"Failed to fetch products. Status code: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error fetching products: {str(e)}")
        return {}

# Function to get user purchases
def get_user_purchases(user_id):
    """
    Gets the purchase history for a user.
    
    :param user_id: The user ID
    :return: List of purchases for the user
    """
    return user_purchases.get(user_id, [])

# Function to add a purchase for a user
def add_user_purchase(user_id, product_id, product_name, quantity, price):
    """
    Adds a purchase to a user's purchase history.
    
    :param user_id: The user ID
    :param product_id: The product ID
    :param product_name: The product name
    :param quantity: The quantity purchased
    :param price: The price of the product
    """
    if user_id not in user_purchases:
        user_purchases[user_id] = []
    
    # Generate a unique order ID that looks like a regular order ID
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

# Function to enrich orders with product details
def enrich_orders_with_product_details(orders, products):
    """
    Enriches order data with product details.
    
    :param orders: List of orders
    :param products: Dictionary of product details
    :return: Enriched orders with product details
    """
    enriched_orders = []
    
    # Get a list of product values to use when IDs don't match
    product_list = list(products.values())
    total_products = len(product_list)
    
    for order in orders:
        enriched_order = order.copy()
        enriched_items = []
        
        # Process each item in the order
        if 'items' in order:
            for item in order['items']:
                product_id = None
                quantity = 1
                
                # Handle different item formats
                if isinstance(item, dict) and 'product_id' in item:
                    product_id = item['product_id']
                    quantity = item.get('quantity', 1)
                elif isinstance(item, int):
                    product_id = item
                
                # Try to find product in our products dictionary
                if product_id in products:
                    product = products[product_id]
                else:
                    # If product ID doesn't exist in our products, assign a random product
                    # Use the product_id as an index (with modulo to stay in bounds)
                    fallback_index = (product_id % total_products) if total_products > 0 else 0
                    product = product_list[fallback_index] if total_products > 0 else {"title": f"Product {product_id}"}
                
                # Create enriched item with product details
                enriched_item = {
                    'product_id': product_id,
                    'name': product.get('title', f'Product {product_id}'),
                    'price': product.get('price', 0),
                    'quantity': quantity,
                    'category': product.get('category', 'Unknown'),
                    'description': product.get('description', '')[:100] + '...' if product.get('description', '') and len(product.get('description', '')) > 100 else product.get('description', ''),
                    'image': product.get('thumbnail', '')
                }
                enriched_items.append(enriched_item)
        
        enriched_order['enriched_items'] = enriched_items
        enriched_orders.append(enriched_order)
    
    return enriched_orders

# Callback handler for streaming tokens with controlled speed
class ChainlitAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # Add a small delay between tokens for a more natural reading pace
        time.sleep(0.01)  # 10ms delay between tokens
        asyncio.run(cl.user_session.get("current_msg").stream_token(token))

# Function to create and initialize the orchestrator with all agents
def create_orchestrator(user_id):
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
           - High: Issues affecting service, urgent technical issues
           - Medium: Non-urgent product inquiries, sales questions
           - Low: General information requests, feedback
        3. Key Entities: Extract important product names or specific issues mentioned.
        4. For follow-up responses, include relevant entities from the previous interaction if applicable.
        5. Confidence: Indicate how confident you are in the classification.
           - High: Clear, straightforward requests or clear follow-ups
           - Medium: Requests with some ambiguity but likely classification
           - Low: Vague or multi-faceted requests that could fit multiple categories

        Company context:
        - TechGadgets Inc. sells electronics and accessories
        - Order-related queries should go to the Order Agent
        - Product info and general questions go to the Query Agent
        - Purchase-related queries should go to the Sales Agent
        
        For short responses like "yes", "ok", "I want to know more", or numerical answers,
        treat them as follow-ups and maintain the previous agent selection.
        """
    )
    
    # Add agents to the orchestrator
    orchestrator.add_agent(create_query_agent())
    orchestrator.add_agent(create_order_agent(user_id))
    orchestrator.add_agent(create_sales_agent(user_id))
    
    return orchestrator

# Create query agent
def create_query_agent():
    # Fetch products for the query agent
    products = fetch_products()
    product_categories = set()
    popular_products = []
    
    # Extract categories and select some popular products
    for product in list(products.values())[:50]:  # Limit to first 20 for popular products
        if 'category' in product:
            product_categories.add(product['category'])
        
        if len(popular_products) < 5 and 'title' in product:
            popular_products.append({
                'name': product['title'],
                'price': product.get('price', 'N/A'),
                'category': product.get('category', 'Unknown'),
                'rating': product.get('rating', 0)
            })
    
    product_info = {
        'categories': list(product_categories),
        'popular_products': popular_products
    }
    
    return OpenAIAgent(OpenAIAgentOptions(
        name='Query Agent',
        description='Specializes in answering customer FAQs, general inquiries, and product information',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        callbacks=ChainlitAgentCallbacks(),
        LOG_AGENT_DEBUG_TRACE=True,
        inference_config={
            'maxTokens': 800,
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
            ''' + json.dumps(product_info, indent=2) + '''
            
            Policies:
            - Return policy: 30-day return window for most products
            - Warranty: Manufacturer warranty applies to all products
            - Shipping: Free shipping on orders over $50
            - Price match: We match prices from major competitors
            
            When answering questions:
            - Be friendly and helpful
            - Provide specific information when available
            - If you don't know the answer, say so and offer to connect the customer with someone who can help
            - For detailed product specifications, recommend checking the product page on our website''',
        }
    ))

# Create order agent
def create_order_agent(user_id):
    
    # Get user purchases - this should have the latest status
    purchases = get_user_purchases(user_id)
    
    # Format orders for the agent
    formatted_orders = []
    
    # Add user purchases as orders
    for purchase in purchases:
        formatted_order = {
            "order_id": purchase["order_id"],
            "date": purchase["date"],
            "status": purchase["status"],  # This should have the updated status
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
        LOG_AGENT_DEBUG_TRACE=True,
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
            
            Customer data including orders:
            ''' + formatted_user_data + '''
            
            When handling order-related queries:
            - If the customer doesn't mention a specific order number, list their recent orders briefly
            - For each order, show the order ID, status, and list the products
            - If they mention a specific order number, provide complete details for that order
            - If the customer has no orders yet, inform them that they haven't made any purchases
            - For order modifications, explain the process clearly
            - Always be helpful and solution-oriented
            
            Order status information:
            - Processing: Order has been received and is being processed
            - Preparing for Shipment: Order is being packed
            - Shipped: Order has been shipped and is in transit
            - Out for Delivery: Order is with the delivery carrier and will be delivered soon
            - Delivered: Order has been delivered
            
            Shipping information:
            - Standard shipping: 3-5 business days
            - Express shipping: 1-2 business days
            - International shipping: 7-14 business days''',
        }
    ))

# Create sales agent
def create_sales_agent(user_id):
    # Fetch products for the sales agent
    products = fetch_products()
    
    # Get user purchases
    purchases = get_user_purchases(user_id)
    
    # Format product data for the sales agent
    formatted_products = []
    for product_id, product in products.items():
        formatted_product = {
            'id': product_id,
            'name': product.get('title', f'Product {product_id}'),
            'price': product.get('price', 0),
            'category': product.get('category', 'Unknown'),
            'description': product.get('description', '')[:100] + '...' if len(product.get('description', '')) > 100 else product.get('description', ''),
            'rating': product.get('rating', 0),
            'stock': product.get('stock', 0),
            'brand': product.get('brand', 'Unknown')
        }
        formatted_products.append(formatted_product)
    
    # Limit to 30 products to avoid token limits
    formatted_products = formatted_products[:30]
    
    return OpenAIAgent(OpenAIAgentOptions(
        name='Sales Agent',
        description='Specializes in product recommendations and purchase assistance',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        callbacks=ChainlitAgentCallbacks(),
        LOG_AGENT_DEBUG_TRACE=True,
        inference_config={
            'maxTokens': 800,
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
            ''' + json.dumps(formatted_products, indent=2) + '''
            
            Customer's previous purchases:
            ''' + json.dumps(purchases, indent=2) + '''
            
            When assisting customers:
            - Ask clarifying questions to understand their needs
            - Recommend products that match their requirements
            - Highlight key features and benefits
            
            
            For purchase requests:
            - When a customer expresses interest in buying a product, ask them how many they would like
            - If they don't specify a quantity, assume they want 1 unit
            - Ask them to confirm their purchase with a simple yes/no question
            - Do NOT ask for payment details as this is a demo system
            - Once they confirm, tell them you're processing their order
            - Provide a purchase confirmation with the product name, quantity, price, and a thank you message
            - Let them know they can check their order status with the Order Agent
            
            Important: This is a demo system, so keep the purchase process simple. Just ask for quantity and confirmation, then complete the purchase.

            Warranty information:
            - All products come with a 1-year manufacturer warranty
            - 30-day satisfaction guarantee''',
        }
    ))

@cl.on_chat_start
async def start():
    # Initialize session variables
    user_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("pending_purchase", None)
    # Send welcome message
    await cl.Message(
        content=f"""# Welcome to TechGadgets Customer Support
        
You're logged in as: {DEFAULT_USER['email']} (Demo User)

### Sample questions you can ask:
- What's your return policy?
- I want to check my order status
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
    
    # Always recreate the orchestrator to get the latest order status
    orchestrator = create_orchestrator(user_id)
    cl.user_session.set("orchestrator", orchestrator)
    
    # Create an empty message for streaming
    msg = cl.Message(content="")
    await msg.send()  # Send the message immediately to start streaming
    cl.user_session.set("current_msg", msg)
    
    # Check for pending purchase intent
    pending_purchase = cl.user_session.get("pending_purchase", None)
    
    # If there's a pending purchase and the user confirms
    if pending_purchase and any(word in message.content.lower() for word in ["yes", "confirm", "sure", "ok", "okay", "proceed", "buy it"]):
        # Add the purchase to the user's history
        product_id = pending_purchase["product_id"]
        product_name = pending_purchase["product_name"]
        quantity = pending_purchase["quantity"]
        price = pending_purchase["price"]
        
        # Create the purchase
        purchase = add_user_purchase(user_id, product_id, product_name, quantity, price)
        
        # Create a new orchestrator with updated agents
        orchestrator = create_orchestrator(user_id)
        cl.user_session.set("orchestrator", orchestrator)
        
        # Clear the pending purchase
        cl.user_session.set("pending_purchase", None)
        
        # Send purchase confirmation with controlled streaming speed
        confirmation = f"""# Purchase Confirmed!

Your order has been successfully processed.

**Order Details:**
- **Order ID:** {purchase['order_id']}
- **Product:** {product_name}
- **Quantity:** {quantity}
- **Price:** ${price}
- **Total:** ${price * quantity}
- **Status:** {purchase['status']}
- **Estimated Delivery:** {purchase['estimated_delivery']}

Thank you for your purchase! You can check your order status anytime by asking about your orders or specifically about order {purchase['order_id']}.

Note: Your order status will automatically update as it progresses through our system.
"""

        # Stream the confirmation with a controlled pace
        for line in confirmation.split('\n'):
            await msg.stream_token(line + '\n')
            await asyncio.sleep(0.1)  # 100ms delay between lines for better readability
        await msg.update()
        return
    
    # Route the request to the appropriate agent
    response = await orchestrator.route_request(message.content, user_id, session_id, {})
    
    # Check if the message contains purchase intent
    if any(keyword in message.content.lower() for keyword in ["buy", "purchase", "order", "get me", "i want", "i'd like"]):
        # Extract product information from the message
        products = fetch_products()
        for product_id, product in products.items():
            product_name = product.get('title', '').lower()
            if product_name.lower() in message.content.lower():
                # Check if quantity is specified in the message
                quantity = 1  # Default quantity
                
                # Look for quantity patterns
                import re
                quantity_match = re.search(r'(\d+)\s+' + re.escape(product_name), message.content.lower())
                if not quantity_match:
                    quantity_match = re.search(r'buy\s+(\d+)', message.content.lower())
                if not quantity_match:
                    quantity_match = re.search(r'get\s+(\d+)', message.content.lower())
                if not quantity_match:
                    quantity_match = re.search(r'order\s+(\d+)', message.content.lower())
                if not quantity_match:
                    quantity_match = re.search(r'purchase\s+(\d+)', message.content.lower())
                    
                if quantity_match:
                    try:
                        quantity = int(quantity_match.group(1))
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
                confirmation_request = f"\n\nI see you're interested in purchasing {quantity} {product.get('title')} at a price of ${product.get('price', 0)} each. Would you like to confirm this purchase? (Just say 'yes' to confirm)"
                await msg.stream_token(confirmation_request)
                break
    
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