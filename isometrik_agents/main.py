import asyncio
import json
import time
from typing import AsyncIterable, Dict, Any, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from threading import Thread
from .isometrik_agent_core import create_orchestrator
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.agents import AgentCallbacks, AgentResponse

app = FastAPI()

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



class MyCustomHandler(AgentCallbacks):
    def __init__(self, queue) -> None:
        super().__init__()
        self._queue = queue
        self._stop_signal = None
        print("Custom handler Initialized")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        time.sleep(0.05)
        self._queue.put_nowait(token)
        print(token, end="", flush=True)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print("generation started")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("\n\ngeneration concluded")
        print("LLM generation complete")


def setup_orchestrator(streamer_queue):
    streaming_handler = MyCustomHandler(streamer_queue)
    orchestrator = create_orchestrator(streaming_handler=streaming_handler)
    
    return orchestrator


async def start_generation(query, user_id, session_id, streamer_queue):
    try:
        orchestrator = setup_orchestrator(streamer_queue)

        response = await orchestrator.route_request(query, user_id, session_id)
        if isinstance(response, AgentResponse) and response.streaming is False:
            if isinstance(response.output, str):
                streamer_queue.put_nowait(response.output)
            elif isinstance(response.output, ConversationMessage):
                streamer_queue.put_nowait(response.output.content[0].get('text'))
    except Exception as e:
        print(f"Error in start_generation: {e}")
    finally:
        streamer_queue.put_nowait(None)  

async def response_generator(query, user_id, session_id):
    streamer_queue = asyncio.Queue()

    Thread(target=lambda: asyncio.run(start_generation(query, user_id, session_id, streamer_queue))).start()

    print("Waiting for the response...")
    while True:
        try:
            try:
                value = await asyncio.wait_for(streamer_queue.get(), 0.1)
                
                if value is None:
                    yield f"data: [DONE]\n\n"
                    break
                
                yield f"data: {value}\n\n"
                
                streamer_queue.task_done()
            except asyncio.TimeoutError:
                continue
        except Exception as e:
            print(f"Error in response_generator: {str(e)}")
            yield f"data: Error: {str(e)}\n\n"
            break


@app.post("/stream_chat/")
async def stream_chat(body: ChatRequest):
    return StreamingResponse(
        response_generator(body.content, body.user_id, body.session_id), 
        media_type="text/event-stream"
    )



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