# Multi-Agent AI Content Orchestrator

A sophisticated content generation system that orchestrates conversations between multiple AI agents to create high-quality LinkedIn posts. The project supports multiple AI providers including OpenAI, Google Gemini, and AWS Bedrock.

## Features

- Multi-agent orchestration for collaborative content generation
- Support for multiple AI providers:
  - Google Gemini
  - OpenAI
  - AWS Bedrock
- Specialized LinkedIn post generation with different implementation approaches
- Automated content storage and organization

## Installation

```bash
pip install -r requirements.txt
```


### Core Components

- `os_model.py` - Core agent implementation using Google's Gemini model
- `linkdn_post_orchestrator.py` - LinkedIn post generator using Gemini API
- `linkedn_with_orch_openai_v2.py` - OpenAI-based implementation with process_request orchestration
- `linkedn_orch_with_openai.py` - OpenAI-based implementation with route_request orchestration and OpenAI classifier (removing AWS Bedrock dependency)

### Output Directories

- `generated_posts/` - Stores posts generated using Gemini implementation
- `openai_posts/` - Stores posts generated using OpenAI implementation

## Usage

### Basic Demo

```bash
python orchestrator_demo.py
```

### OpenAI Implementation

```bash
python openai_demo.py
```

## Generated Content Examples

The system has successfully generated posts on various technical topics including:
- Developer Productivity
- OpenAI vs Meta Comparisons
- Google Gemini Analysis
- Kubernetes
- OpenAI vs Anthropic Comparisons

## Configuration

Before running the project, ensure you have:
1. Required API keys set up for your chosen AI provider
2. Proper environment variables configured
3. Necessary permissions if using AWS Bedrock

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

