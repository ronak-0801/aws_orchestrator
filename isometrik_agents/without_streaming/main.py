import uuid
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import json
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
                response_text = response.output.content[0].get("text", "")
                # Check for products in the response
                products = response.output.content[0].get("products", [])
                if products:
                    # If products are present, format response with products widget
                    return {
                        "response": "We've found exactly what you're looking for! Check out your search results below.",
                        "request_id": str(uuid.uuid4()),
                        "widgets": [
                            {
                                "widgetId": 2,
                                "widgets_type": 2,
                                "type": "products",
                                "widget": products
                            },
                            {
                                "widgetId": 1,
                                "widgets_type": 1,
                                "type": "options",
                                "widget": ["Load More Products"]
                            }
                        ]
                    }
                
                # Apply regex extraction for text and options
                if isinstance(response_text, str) and '"text"' in response_text and '"options"' in response_text:
                    text_match = re.search(r'"text":\s*"([^"]*)"', response_text)
                    extracted_text = text_match.group(1) if text_match else ""
                    
                    options_match = re.search(r'"options":\s*\[(.*?)\]', response_text)
                    if options_match:
                        options_str = options_match.group(1)
                        extracted_options = [opt.strip('"') for opt in re.findall(r'"([^"]*)"', options_str)]
                    else:
                        extracted_options = []
                    
                    # Format response with widgets structure
                    return {
                        "response": extracted_text,
                        "request_id": str(uuid.uuid4()),
                        "widgets": [
                            {
                                "widgetId": 1,
                                "widgets_type": 1,
                                "type": "options",
                                "widget": extracted_options
                            }
                        ]
                    }
                
                # Default handling if regex patterns don't match
                return response.output
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
