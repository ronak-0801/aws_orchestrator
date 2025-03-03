# Multi-Agent Orchestrator

This is a multi-agent orchestrator that uses the Gemini API to orchestrate a conversation between a user and a group of agents.

## Installation

```bash
pip install -r requirements.txt
```

## Usage


```bash
python orchestrator_demo.py
```

## Project Files

- `os_model.py` - Custom agent implementation using Google's Gemini model for AI responses
- `linkdn_post_orchestrator.py` - LinkedIn post generation using multi-agent orchestration with levaraging Gemini API instead using aws Bedrock
- `linkedn_with_orch_openai_v2.py` - LinkedIn post generation using OpenAI API with orchestrator using process_request and not depending on aws bedrock
- `linkedn_orch_with_openai.py` - LinkedIn post generation using OpenAI API with orchestrator using route_request and depending on aws bedrock