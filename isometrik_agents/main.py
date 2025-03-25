import asyncio
import time
import queue
from typing import AsyncIterable, List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .isometrik_agent_core import create_orchestrator
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.agents import AgentResponse, AgentCallbacks

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

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: 
            raise item
        return item

    def send(self, data):
        # Send all data, even empty strings
        self.queue.put(data)

    def set_res_dict(self, res_dict):
        self.res_dict = res_dict

    def close(self):
        self.queue.put(StopIteration)

class StreamingHandler(AgentCallbacks):
    def __init__(self, generator: ThreadedGenerator) -> None:
        super().__init__()
        self.generator = generator
        self.stream_finished = False
        print("Streaming handler initialized")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if not self.stream_finished:
            time.sleep(0.05)
            self.generator.send(token)
            print(token, end="", flush=True) 

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("\nGeneration started")
        self.stream_finished = False
        self.generator.send(" ")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("\nGeneration concluded")
        if not self.stream_finished:
            self.generator.send("\n")
            self.generator.close()
            self.stream_finished = True

async def agent_stream_response(request: Request, generator: ThreadedGenerator):
    """Convert the threaded generator to an async generator for StreamingResponse."""
    try:
        # Send a start event
        yield "event: start\ndata: Stream started\n\n"
        

        # Create an iterator from the generator
        iterator = iter(generator)
        
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                print("Client disconnected")
                break
                

            try:
                # Get the next token from the generator with a timeout
                token = await asyncio.to_thread(next, iterator)
                
                # Properly format the SSE message
                # Escape any newlines in the token
                escaped_token = token.replace("\n", "\\n") if isinstance(token, str) else str(token)
                yield f"event: token\ndata: {escaped_token}\n\n"
                
            except StopIteration:
                # End of stream
                print("End of stream")
                yield "event: end\ndata: Stream ended\n\n"
                break
            except Exception as e:
                print(f"Error getting next token: {str(e)}")
                yield f"event: error\ndata: Error getting next token: {str(e)}\n\n"
                break
                
            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"Stream error: {str(e)}")
        yield f"event: error\ndata: {str(e)}\n\n"
    finally:
        print("Stream generator finished")

@app.post("/stream_chat/")
async def chat(request: Request, body: ChatRequest):
    try:
        # Create a threaded generator for streaming
        generator = ThreadedGenerator()
        
        # Create a streaming handler with the generator
        handler = StreamingHandler(generator)
        
        # Create orchestrator with streaming handler
        orchestrator = create_orchestrator(streaming_handler=handler)
        
        # Start processing in a background task
        asyncio.create_task(
            orchestrator.route_request(
                body.content,
                body.user_id,
                body.session_id
            )
        )
        
        # Return a streaming response
        return StreamingResponse(
            agent_stream_response(request, generator),
            media_type="text/event-stream",

        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)