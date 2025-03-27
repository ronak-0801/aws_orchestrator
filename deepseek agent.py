from typing import List, Optional, Dict, Any
import os
import openai
from multi_agent_orchestrator.agents import Agent, AgentOptions
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator
import asyncio
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions


class DeepSeekAgentOptions(AgentOptions):
    """Configuration options for the DeepSeek Agent."""
    # api_key: str = "sk-854a75b31f744899af227aa6d92f158f"
    # model: str = Field(default="deepseek-chat")  # Default model - uses DeepSeek-V3
    # api_base: str = Field(default="https://api.deepseek.com")  # Base URL without v1 suffix
    # temperature: float = Field(default=0.7)
    # max_tokens: int = Field(default=1000)


class DeepSeekAgent(Agent):

    def __init__(self, options: DeepSeekAgentOptions):
        super().__init__(options)
        # Set API key and base URL for the OpenAI client
        self.client = openai.OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )


    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": input_text}],
            stream=False
        )
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": response.choices[0].text.strip()}]
        )



# Load your API key from environment variable
api_key = os.getenv("DEEPSEEK_API_KEY")

custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
    api_key=api_key,
    model_id="deepseek-reasoner",
))
# Initialize the orchestrator
orchestrator = MultiAgentOrchestrator(
    classifier=custom_openai_classifier
)

# Create the DeepSeek agent
deepseek_agent = DeepSeekAgent(
    DeepSeekAgentOptions(
        name="DeepSeek Assistant",
        description="An agent that uses DeepSeek's language models for responses",
    )
)

# Add the agent to the orchestrator
orchestrator.add_agent(deepseek_agent)

# Test function
async def test_agent():
    response = await orchestrator.route_request(
        "Tell me about artificial intelligence",
        user_id="test_user",
        session_id="test_session",
    )
    breakpoint()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_agent())
