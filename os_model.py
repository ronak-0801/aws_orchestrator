import os
import openai
from typing import List, Optional, Dict
from multi_agent_orchestrator.agents import Agent, AgentOptions
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
import requests
from dotenv import load_dotenv


load_dotenv()
# Set your Gemini API key (replace with actual key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


class GeminiAgentOptions(AgentOptions):
    """Options for GeminiAgent configuration"""
    def __init__(self, name: str, description: str, api_key: str):
        super().__init__(name=name, description=description)
        self.api_key = api_key

class GeminiAgent(Agent):
    """Custom agent that uses Google's Gemini model"""
    
    def __init__(self, options: GeminiAgentOptions):
        super().__init__(options)
        self.api_key = options.api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: List[ConversationMessage],
        additional_params: Optional[Dict[str, str]] = None
    ) -> ConversationMessage:
        """Process the incoming request and return a response"""
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Format the request payload according to Gemini API specs
            data = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": input_text}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                }
            }

            # Add API key to URL instead of params
            url = f"{self.api_url}?key={self.api_key}"
            
            response = requests.post(
                url,
                headers=headers,
                json=data
            )
            
            print(f"Status Code: {response.status_code}")  # Debug line
            print(f"Response: {response.text}")  # Debug line
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                response_text = f"API Error: {response.status_code} - {response.text}"

            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": response_text}]
            )
            
        except Exception as e:
            print(f"Exception details: {str(e)}")  # Debug line
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": f"Error processing request: {str(e)}"}]
            )

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        # Initialize the agent
        gemini_agent = GeminiAgent(GeminiAgentOptions(
            name='Gemini Agent',
            description='An agent that uses Google Gemini 2.0 Flash model for responses',
            api_key=os.getenv("GEMINI_API_KEY")
        ))

        # Test question
        response = await gemini_agent.process_request(
            input_text="What is quantum computing?",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )
        
        print("Response:", response.content[0]["text"])

    # Run the test
    asyncio.run(test_agent())
