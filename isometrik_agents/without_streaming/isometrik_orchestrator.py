import asyncio
import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions
from multi_agent_orchestrator.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorConfig,
)
from multi_agent_orchestrator.classifiers import (
    OpenAIClassifier,
    OpenAIClassifierOptions,
)
from multi_agent_orchestrator.storage import InMemoryChatStorage
from multi_agent_orchestrator.retrievers import Retriever

load_dotenv()

memory_storage = InMemoryChatStorage()

custom_system_prompt = {
                "template": """CRITICAL: You are a JSON pass-through agent.
Your only task is to return the exact JSON context provided, with zero modifications.
Do not add text, do not modify structure, do not format anything and dont add any other text.
Return the raw JSON context exactly as provided.
"""
}


class IsometrikRetrieverOptions:
    def __init__(self, endpoint: str, auth_token: str, agent_id: str, session_id: str):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.agent_id = agent_id
        self.session_id = session_id


class IsometrikRetriever(Retriever):
    def __init__(self, options: IsometrikRetrieverOptions):
        super().__init__(options)
        self.options = options

        if not all([options.endpoint, options.auth_token, options.agent_id]):
            raise ValueError(
                "endpoint, auth_token, and agent_id are required in options"
            )

    async def retrieve_and_generate(
        self, text, retrieve_and_generate_configuration=None
    ):
        pass

    async def retrieve(self, text: str) -> List[Dict[str, Any]]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.options.auth_token}",
            }

            payload = {
                "session_id": self.options.session_id, 
                "message": text,
                "agent_id": self.options.agent_id,
            }

            response = await asyncio.to_thread(
                requests.post, self.options.endpoint, headers=headers, json=payload
            )

            if response.status_code != 200:
                print(f"Retriever error: HTTP {response.status_code} - {response.text}")
                raise Exception(f"HTTP error! status: {response.status_code}")
            
            
            content = response.text
            data = json.loads(content)
            print(data)
            return data

        except Exception as e:
            print(f"Retriever error: {str(e)}")
            raise Exception(f"Failed to retrieve: {str(e)}")

    async def retrieve_and_combine_results(self, text: str) -> Dict[str, Any]:

        results = await self.retrieve(text)
        return self._combine_results(results)

    @staticmethod
    def _combine_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:

        return results["text"]


def create_query_retriever():
    return IsometrikRetriever(
        IsometrikRetrieverOptions(
            endpoint=os.getenv("QUERY_AGENT_API_URL"),
            auth_token=os.getenv("QUERY_AGENT_AUTH_TOKEN"),
            agent_id=os.getenv("QUERY_AGENT_ID"),
            session_id=os.getenv("QUERY_AGENT_SESSION_ID"),
        )
    )


def create_order_retriever():
    return IsometrikRetriever(
        IsometrikRetrieverOptions(
            endpoint=os.getenv("ORDER_AGENT_API_URL"),
            auth_token=os.getenv("ORDER_AGENT_AUTH_TOKEN"),
            agent_id=os.getenv("ORDER_AGENT_ID"),
            session_id=os.getenv("ORDER_AGENT_SESSION_ID"),
        )
    )


def create_ecom_manager_retriever():
    return IsometrikRetriever(
        IsometrikRetrieverOptions(
            endpoint=os.getenv("MANAGER_AGENT_API_URL"),
            auth_token=os.getenv("MANAGER_AGENT_AUTH_TOKEN"),
            agent_id=os.getenv("MANAGER_AGENT_ID"),
            session_id=os.getenv("MANAGER_AGENT_SESSION_ID"),
        )
    )


def create_subscription_retriever():
    return IsometrikRetriever(
        IsometrikRetrieverOptions(
            endpoint=os.getenv("SUBSCRIPTION_AGENT_API_URL"),
            auth_token=os.getenv("SUBSCRIPTION_AGENT_AUTH_TOKEN"),
            agent_id=os.getenv("SUBSCRIPTION_AGENT_ID"),
            session_id=os.getenv("SESSION_ID"),
        )
    )


def create_product_retriever():
    return IsometrikRetriever(
        IsometrikRetrieverOptions(
            endpoint=os.getenv("PRODUCT_AGENT_API_URL"),
            auth_token=os.getenv("PRODUCT_AGENT_AUTH_TOKEN"),
            agent_id=os.getenv("PRODUCT_AGENT_ID"),
            session_id=os.getenv("PRODUCT_AGENT_SESSION_ID"),
        )
    )


def create_query_agent():
    return OpenAIAgent(
        OpenAIAgentOptions(
            name="Query Agent",
            description="Specializes in answering customer FAQs, general inquiries, and product information",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=False,
            retriever=create_query_retriever(),
            custom_system_prompt=custom_system_prompt
        )
    )


def create_order_agent():
    return OpenAIAgent(
        OpenAIAgentOptions(
            name="Order Agent",
            description="Specializes in handling order-related queries, order status, and processing new orders",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=False,
            retriever=create_order_retriever(),
            custom_system_prompt=custom_system_prompt
        )
    )


def create_manager_agent():
    return OpenAIAgent(
        OpenAIAgentOptions(
            name="Eco Manager Agent",
            description="Specializes in finding toxin-free, eco-friendly solutions for home and personal care",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=False,
            retriever=create_ecom_manager_retriever(),
            custom_system_prompt=custom_system_prompt
        )
    )


def create_subscription_agent():
    return OpenAIAgent(
        OpenAIAgentOptions(
            name="Subscription Agent",
            description="Specializes in handling subscription-related queries",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=False,
            retriever=create_subscription_retriever(),
            custom_system_prompt=custom_system_prompt
        )
    )


def create_product_agent():
    return OpenAIAgent(
        OpenAIAgentOptions(
            name="Product Details Agent",
            description="Specializes in providing detailed product information, recommendations, comparisons, and purchase advice.",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=False,
            retriever=create_product_retriever(),
            custom_system_prompt=custom_system_prompt
        )
    )


def create_orchestrator():
    """Creates and initializes the orchestrator with all agents with streaming support."""
    custom_openai_classifier = OpenAIClassifier(
        OpenAIClassifierOptions(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_id="gpt-4o-mini",
        )
    )

    orchestrator = MultiAgentOrchestrator(
        options=OrchestratorConfig(
            LOG_CLASSIFIER_OUTPUT=True,
            LOG_CLASSIFIER_RAW_OUTPUT=True,
            MAX_RETRIES=2,
            USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
            MAX_MESSAGE_PAIRS_PER_AGENT=8,
        ),
        classifier=custom_openai_classifier,
        # storage=memory_storage,
    )

    query_agent = create_query_agent()
    order_agent = create_order_agent()
    manager_agent = create_manager_agent()
    subscription_agent = create_subscription_agent()
    product_agent = create_product_agent()

    orchestrator.add_agent(query_agent)
    orchestrator.add_agent(order_agent)
    orchestrator.add_agent(manager_agent)
    orchestrator.add_agent(subscription_agent)
    orchestrator.add_agent(product_agent)

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
           - Query Agent: General product information including greetings, FAQs, and basic inquiries
           - Order Agent: Order status, tracking, processing, and shipping
           - Eco Manager Agent: Eco-friendly products, toxin-free solutions, sustainable living
           - Subscription Agent: Subscription-related queries
           - Product Details Agent: Detailed product information, recommendations, comparisons, and purchase advice.

        2. Priority: Assign based on urgency and impact.
           - High: Issues affecting service, urgent requests, or critical product information.
           - Medium: Non-urgent product inquiries, general questions, or subscription management.
           - Low: Information requests, browsing, or vague inquiries.

        3. Key Entities: Extract important product names, issues, or specific requests mentioned.
           For follow-ups, include relevant entities from previous interactions.

        4. Confidence: Indicate how confident you are in the classification.
           - High: Clear, straightforward requests or clear follow-ups.
           - Medium: Requests with some ambiguity but likely classification.
           - Low: Vague or multi-faceted requests that could fit multiple categories.

        5. Is Followup: Indicate whether the input is a follow-up to a previous interaction.

        Specific Classification Rules:
        - If the user asks about general information or product searches or perticular product or category → Query Agent.
        - If the user asks about order status, tracking, or shipping, past orders → Order Agent.
        - If the user asks about subscription management, billing, or account status → Subscription Agent.
        - If the user asks for detailed product specifications, recommendations, comparisons, or purchase advice , main target is product or category recommendations → Product Details Agent.
        - If the user asks about eco-friendly alternatives, sustainable products, or environmental impact → Eco Manager Agent.

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
