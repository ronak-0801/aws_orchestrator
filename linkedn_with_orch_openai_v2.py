# This is a openai implementation with using process_request and not depending on aws bedrock


from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from multi_agent_orchestrator.classifiers import ClassifierResult

load_dotenv()

def create_research_agent():
    """Creates an agent specialized in research and content planning"""
    return OpenAIAgent(OpenAIAgentOptions(
        name='Research Agent',
        description='Specializes in researching topics and creating content outlines',
        api_key=os.getenv('OPENAI_API_KEY'),
        custom_system_prompt={
            'template': '''You are an expert researcher specialized in {{DOMAIN}}. 
            Your task is to analyze topics and provide key insights, trends, and supporting data.
            Return your response in the following JSON format only (no other text):
            {
                "key_points": ["point 1", "point 2", "point 3"],
                "relevant_stats": ["stat 1", "stat 2"],
                "target_audience": "description of target audience",
                "suggested_hashtags": ["hashtag1", "hashtag2", "hashtag3"]
            }''',
            'variables': {
                'DOMAIN': 'business, technology, and industry trends'
            }
        }
    ))

def create_content_writer_agent():
    """Creates an agent specialized in writing engaging content"""
    return OpenAIAgent(OpenAIAgentOptions(
        name='Content Writer',
        description='Specializes in crafting engaging LinkedIn posts',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        inference_config={
            'maxTokens': 1000,
            'temperature': 0.8
        },
        custom_system_prompt={
            'template': '''You are an expert content writer specialized in {{DOMAIN}}. 
            Create engaging content that incorporates the research provided.
            Focus on {{STYLE}} while maintaining professionalism.''',
            'variables': {
                'DOMAIN': 'LinkedIn post writing',
                'STYLE': 'storytelling, personal insights, and value-driven content'
            }
        }
    ))

def create_engagement_optimizer_agent():
    """Creates an agent specialized in optimizing posts for LinkedIn engagement"""
    return OpenAIAgent(OpenAIAgentOptions(
        name='Engagement Optimizer',
        description='Specializes in optimizing content for LinkedIn engagement',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        inference_config={
            'maxTokens': 1000,
            'temperature': 0.6
        },
        custom_system_prompt={
            'template': '''You are an expert in {{DOMAIN}}. 
            Format the content in clean markdown format.
            Use proper markdown syntax for:
            - Headers (##)
            - Lists (-)
            - Bold text (**)
            - Emphasis (_)
            - Line breaks
            
            Do not include any raw JSON or list structures in the output.
            Return only the formatted markdown content.
            
            Ensure the post follows LinkedIn best practices for formatting and length.
            ''',
            'variables': {
                'DOMAIN': 'LinkedIn content optimization and markdown formatting',
                'FOCUS': 'clean markdown structure and readability'
            }
        }
    ))

async def generate_linkedin_post(topic):
    orchestrator = MultiAgentOrchestrator()
    research_agent = create_research_agent()
    orchestrator.add_agent(research_agent)
    
    # Step 1: Research Phase
    print("\n1. Researching the topic...")
    classifier_result = ClassifierResult(selected_agent=research_agent, confidence=1.0)
    
    research_response = await orchestrator.agent_process_request(
        f"Analyze this topic and provide research for a LinkedIn post: {topic}",
        "linkedin_generator",
        "post_gen",
        classifier_result
    )
    
    try:
        content = research_response.output.content[0]['text']
        research_data = json.loads(content)
        print("\nResearch data:", research_data)
        
    except Exception as e:
        print(f"\nError processing research data: {e}")
        # Fallback data generation...
        research_data = {
            "key_points": [f"Key insights about {topic}"],
            "relevant_stats": ["Industry statistics"],
            "target_audience": "Professionals interested in " + topic,
            "suggested_hashtags": ["#Technology", "#Innovation"]
        }
    

    # Step 2: Content Writing Phase
    content_agent = create_content_writer_agent()
    orchestrator.add_agent(content_agent)
    classifier_result = ClassifierResult(selected_agent=content_agent, confidence=1.0)
    
    writing_prompt = f"""
    Create a LinkedIn post about {topic} using these insights:
    Key Points: {research_data['key_points']}
    Stats: {research_data['relevant_stats']}
    Target Audience: {research_data['target_audience']}
    """
    
    draft_response = await orchestrator.agent_process_request(
        writing_prompt,
        "linkedin_generator",
        "post_gen",
        classifier_result
    )
    
    draft_content = draft_response.output.content[0]['text']

    # Step 3: Optimization Phase
    optimizer_agent = create_engagement_optimizer_agent()
    orchestrator.add_agent(optimizer_agent)
    classifier_result = ClassifierResult(selected_agent=optimizer_agent, confidence=1.0)
    
    optimization_prompt = f"""
    Format this LinkedIn post in markdown:
    
    Topic: {topic}
    Content: {draft_content}
    Hashtags: {research_data['suggested_hashtags']}
    """
    
    final_response = await orchestrator.agent_process_request(
        optimization_prompt,
        "linkedin_generator",
        "post_gen",
        classifier_result
    )
    
    final_content = final_response.output.content[0]['text']
    
    # Format and save the post
    markdown_post = f"""# {topic}

{final_content}

---
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save to file
    os.makedirs('openai_posts', exist_ok=True)
    filename = f"openai_posts/{topic.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown_post)
    
    print(f"\nPost saved to: {filename}")
    return markdown_post

async def main():
    topic = input("Enter a topic for the LinkedIn post: ")
    final_post = await generate_linkedin_post(topic)
    print("\nFinal LinkedIn Post:")
    print("="*50)
    print(final_post)
    print("="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
