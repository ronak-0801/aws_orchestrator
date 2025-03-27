from typing import List, Optional, Dict, Any
import os
# import openai
from openai import OpenAI
from multi_agent_orchestrator.agents import Agent, AgentOptions
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator
import asyncio
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions
from groq import Groq
from multi_agent_orchestrator.classifiers import Classifier
from multi_agent_orchestrator.types import ConversationMessage
from typing import List

class DeepSeekAgentOptions(AgentOptions):
    """Configuration options for the DeepSeek Agent."""
    # model: str = Field(default="deepseek-chat")  # Default model - uses DeepSeek-V3
    # api_base: str = Field(default="https://api.deepseek.com")  # Base URL without v1 suffix
    # temperature: float = Field(default=0.7)
    # max_tokens: int = Field(default=1000)


class DeepSeekAgent(Agent):

    def __init__(self, options: DeepSeekAgentOptions):
        super().__init__(options)
        # Set API key and base URL for the OpenAI client
        self.client =Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
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
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": input_text}],
        )
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": response.choices[0].message.content}]
        )



# Load your API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
    api_key=api_key,
    # model_id="deepseek-reasoner",
))

# Create the DeepSeek agent
deepseek_agent = DeepSeekAgent(
    DeepSeekAgentOptions(
        name="DeepSeek Assistant",
        description="An agent that uses DeepSeek's language models for responses",
    )
)
orchestrator = MultiAgentOrchestrator(default_agent=deepseek_agent , classifier=custom_openai_classifier)

# Add the agent to the orchestrator
orchestrator.add_agent(deepseek_agent)

async def test_agent():
    response = await orchestrator.route_request(
        "Hii",
        user_id="test_user",
        session_id="test_session",
    )
    # print(response)
    print(response.output.content[0]['text'])

# Run the test
if __name__ == "__main__":
    asyncio.run(test_agent())
