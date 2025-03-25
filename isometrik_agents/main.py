import asyncio
import json
import time
from typing import AsyncIterable, Dict, Any, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .isometrik_agent_core import create_orchestrator
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.agents import AgentCallbacks, AgentResponse

# Create FastAPI app
app = FastAPI()

# Configure CORS
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

class AsyncStreamingHandler(AgentCallbacks):
    def __init__(self):
        super().__init__()
        self.token_queue = asyncio.Queue()
        self.stream_complete = asyncio.Event()
        self.error_event = asyncio.Event()
        self.error_message = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Add each new token to the async queue"""
        time.sleep(0.5)
        self.token_queue.put_nowait(token)
        print(token, end="", flush=True)  # Print to console for debugging

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Reset events when generation starts"""
        print("\nGeneration started")
        self.stream_complete.clear()
        self.error_event.clear()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Signal stream completion"""
        print("\nGeneration concluded")
        self.token_queue.put_nowait(None)  # Sentinel value
        self.stream_complete.set()

    async def stream_tokens(self) -> AsyncIterable[str]:
        """
        Async generator to stream tokens with proper event handling
        """
        try:
            while not self.stream_complete.is_set():
                # Wait for next token with timeout
                try:
                    token = await asyncio.wait_for(self.token_queue.get(), timeout=10.0)
                    
                    # Check for stream end sentinel
                    if token is None:
                        break
                    
                    # Yield each token
                    yield token
                    
                    # Mark task as done
                    self.token_queue.task_done()
                
                except asyncio.TimeoutError:
                    # Handle potential streaming timeout
                    print("Token timeout reached")
                    break
            
            # Check if an error occurred
            if self.error_event.is_set():
                raise RuntimeError(self.error_message or "Unknown streaming error")
        
        except Exception as e:
            print(f"Error in stream_tokens: {str(e)}")
            yield f"Error: {str(e)}"
        finally:
            # Ensure stream is marked as complete
            self.stream_complete.set()

async def error_event_generator(error_message: str):
    """Generate error events for streaming"""
    yield f"event: error\ndata: {error_message}\n\n"

@app.post("/stream_chat/")
async def stream_chat(request: Request, body: ChatRequest):
    """
    Streaming chat endpoint with robust error handling
    """
    try:
        # Create streaming handler
        streaming_handler = AsyncStreamingHandler()
        
        # Create orchestrator with streaming handler
        orchestrator = create_orchestrator(streaming_handler=streaming_handler)
        
        # Process request asynchronously
        asyncio.create_task(
            orchestrator.route_request(
                body.content, 
                body.user_id, 
                body.session_id
            )
        )
        
        # Create SSE streaming response
        async def event_generator():
            # Send start event
            yield "event: start\ndata: Streaming initiated\n\n"
            
            try:
                # Stream tokens
                async for token in streaming_handler.stream_tokens():
                    # Properly escape token for SSE
                    escaped_token = token.replace("\n", "\\n") if isinstance(token, str) else str(token)
                    yield f"event: token\ndata: {escaped_token}\n\n"
                
                # Send completion event
                yield "event: end\ndata: Stream completed\n\n"
            
            except Exception as e:
                print(f"Error in event_generator: {str(e)}")
                # Error event
                yield f"event: error\ndata: {str(e)}\n\n"
        
        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
        )
    
    except Exception as e:
        print(f"Error in stream_chat endpoint: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000
    )