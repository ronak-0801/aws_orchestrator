import asyncio
import time
from typing import  Dict, Any, List
import re
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Thread
from .isometrik_orchestrator import create_orchestrator
from multi_agent_orchestrator.agents import AgentCallbacks, AgentStreamResponse

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


def setup_orchestrator():
    # streaming_handler = MyCustomHandler(streamer_queue)
    orchestrator = create_orchestrator()
    
    return orchestrator


async def start_generation(query, user_id, session_id):
    try:
        orchestrator = setup_orchestrator()

        response = await orchestrator.route_request(query, user_id, session_id,None,stream_response=True)

        if response.streaming:
            async for chunk in response.output:
                if isinstance(chunk, AgentStreamResponse):
                    print(chunk.text, flush=True, end="")
                    yield f"data: {chunk.text}\n\n"

    except Exception as e:
        print(f"Error in start_generation: {e}")

    # finally:
    #     streamer_queue.put_nowait(None)  

# async def response_generator(query, user_id, session_id):
#     widget =                             {
#                                 "widgetId": 1,
#                                 "widgets_type": 1,
#                                 "type": "options",
#                                 "widget": ""
#                             }
                        
#     streamer_queue = asyncio.Queue()
#     options_list = []
#     collecting_options = False
#     start_stream = False

#     Thread(target=lambda: asyncio.run(start_generation(query, user_id, session_id, streamer_queue))).start()

#     print("Waiting for the response...")
#     while True:
#         try:
#             try:
#                 value = await asyncio.wait_for(streamer_queue.get(), 0.1)
                
#                 if value == "options":
#                     collecting_options = True
#                     continue
#                 elif value is None:
#                     options_text = ''.join(options_list)
#                     matches = re.search(r'\[(.*?)\]', options_text)
#                     if matches:
#                         options = [
#                             opt.strip().strip('"').strip()
#                             for opt in matches.group(1).split(',')
#                         ]
#                         widget["widget"] = options
#                         yield f"data: {widget}\n\n"
#                     break
                
#                 if collecting_options:
#                     options_list.append(value)
#                 else:
#                     if value == '":':
#                         start_stream = True
#                         continue
#                     if start_stream:
#                         print(value, end="", flush=True)
#                         yield f"data: {value}\n\n"
                
#                 streamer_queue.task_done()
#             except asyncio.TimeoutError:
#                 continue
#         except Exception as e:
#             print(f"Error in response_generator: {str(e)}")
#             yield f"data: Error: {str(e)}\n\n"
#             break
    



@app.post("/stream_chat/")
async def stream_chat(body: ChatRequest):
    return StreamingResponse(
        start_generation(body.content, body.user_id, body.session_id), 
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