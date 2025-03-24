import asyncio
from typing import AsyncIterable
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Any

from isometrik_agent_core import create_orchestrator
from multi_agent_orchestrator.agents import AgentResponse, AgentCallbacks
from multi_agent_orchestrator.types import ConversationMessage

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    content: str
    user_id: str
    session_id: str

class StreamingHandler(AgentCallbacks):
    def __init__(self) -> None:
        super().__init__()
        print("Streaming handler initialized")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True) 
        # Print tokens to console in real-time

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("\nGeneration started")


    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("\nGeneration concluded")

@app.post("/chat/")
async def chat(body: ChatRequest):
    try:
        # Create a streaming handler
        handler = StreamingHandler()
        
        # Create orchestrator with streaming handler
        orchestrator = create_orchestrator(streaming_handler=handler)
        
        # Get response
        response = await orchestrator.route_request(
            body.content,
            body.user_id,
            body.session_id
        )
        
        return {"status": "success", "message": "Response generated"}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)