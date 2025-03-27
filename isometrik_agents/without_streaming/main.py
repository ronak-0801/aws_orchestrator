import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .isometrik_orchestrator import create_orchestrator
from multi_agent_orchestrator.types import ConversationMessage
from multi_agent_orchestrator.agents import AgentResponse

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


def setup_orchestrator():
    orchestrator = create_orchestrator()

    return orchestrator


async def start_generation(query, user_id, session_id):
    try:
        orchestrator = setup_orchestrator()
        response = await orchestrator.route_request(query, user_id, session_id)

        if isinstance(response, AgentResponse):
            if isinstance(response.output, ConversationMessage):
                text = response.output.content[0].get("text", "")
                return {
                    "text": text,
                    # 'options': options
                }
            return response.output
        return response
    except Exception as e:
        print(f"Error in start_generation: {e}")
        return {"error": str(e)}


@app.post("/stream_chat/")
async def stream_chat(body: ChatRequest):
    response = await start_generation(body.content, body.user_id, body.session_id)
    return response


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
