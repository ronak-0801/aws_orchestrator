import os
from multi_agent_orchestrator.classifiers import OpenAIClassifier, OpenAIClassifierOptions

from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator
from dotenv import load_dotenv

load_dotenv()

# First agent for general queries
general_agent = OpenAIAgent(OpenAIAgentOptions(
    name='General Assistant',
    description='An assistant for general knowledge queries',
    api_key=os.getenv('OPENAI_API_KEY'),
    custom_system_prompt={
        'template': 'You are an AI assistant specialized in {{DOMAIN}}. Always use a {{TONE}} tone.',
        'variables': {
            'DOMAIN': 'general knowledge',
            'TONE': 'friendly and helpful'
        }
    }
))

# Second agent for technical queries
technical_agent = OpenAIAgent(OpenAIAgentOptions(
    name='Technical Assistant',
    description='An assistant for technical and programming questions',
    api_key=os.getenv('OPENAI_API_KEY'),
    custom_system_prompt={
        'template': 'You are an AI assistant specialized in {{DOMAIN}}. Provide {{STYLE}} responses.',
        'variables': {
            'DOMAIN': 'programming and technology',
            'STYLE': 'detailed technical'
        }
    }
))

custom_openai_classifier = OpenAIClassifier(OpenAIClassifierOptions(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_id='gpt-4o-mini',
    inference_config={
        'max_tokens': 500,
        'temperature': 0.7,
        'top_p': 0.9,
        'stop_sequences': ['']
    }
))



async def main():
    orchestrator = MultiAgentOrchestrator(classifier=custom_openai_classifier)
    
    # Add both agents to the orchestrator
    orchestrator.add_agent(general_agent)
    orchestrator.add_agent(technical_agent)

    # Test questions for both agents
    questions = [
        ("What is the capital of France?", general_agent),
        ("Explain Python decorators", technical_agent)
    ]

    for question, agent in questions:
        print(f"\nQuestion: {question}")
        print(f"Using agent: {agent.name}")
        
        
        response = await orchestrator.route_request(
            question,
            user_id="user123",
            session_id="session456",
            # classifier_result  # Uncomment if you want to force specific agent
        )
        
        # Extract and print the response
        answer = response.output.content[0]['text']
        print(f"Response: {answer}\n")
        print("="*50)
        
        # Add a small delay between requests
        # await asyncio.sleep(2)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())