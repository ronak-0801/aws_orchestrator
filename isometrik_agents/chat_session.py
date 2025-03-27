import asyncio
from typing import List, Dict
from without_streaming.isometrik_orchestrator import create_orchestrator
from multi_agent_orchestrator.agents import AgentResponse
from multi_agent_orchestrator.types import ConversationMessage

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
            self.messages.append({"role": "user", "content": message})
            
            print(f"Routing request to orchestrator: {message}")
            response = await self.orchestrator.route_request(
                message, 
                self.user_id, 
                self.session_id
            )
            
            if isinstance(response, AgentResponse):
                if isinstance(response.output, ConversationMessage):
                    if response.output.content and len(response.output.content) > 0:
                        response_text = response.output.content[0].get("text", "")
                    else:
                        response_text = "No content in response"
                else:
                    response_text = str(response.output)
            else:
                response_text = str(response)
                        
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

if __name__ == "__main__":
    asyncio.run(main()) 