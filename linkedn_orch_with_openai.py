# This is a openai implementation with using route_request and depending on aws bedrock

from multi_agent_orchestrator.agents import OpenAIAgent, OpenAIAgentOptions
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

def create_research_agent():
    """Creates an agent specialized in research and content planning"""
    return OpenAIAgent(OpenAIAgentOptions(
        name='Research Agent',
        description='Specializes in researching topics and creating content outlines',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        streaming=True,
        inference_config={
            'maxTokens': 1000,
            'temperature': 0.7
        },
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
    
    # Add all agents to the orchestrator
    orchestrator.add_agent(create_research_agent())
    orchestrator.add_agent(create_content_writer_agent())
    orchestrator.add_agent(create_engagement_optimizer_agent())
    
    # Step 1: Research Phase
    print("\n1. Researching the topic...")
    research_response = await orchestrator.route_request(
        f"Analyze this topic and provide research for a LinkedIn post. Return a JSON object with key_points, relevant_stats, target_audience, and suggested_hashtags: {topic}",
        user_id="linkedin_generator",
        session_id="post_gen"
    )
    
    try:
        # Get the content and debug print
        content = research_response.output.content
        print("\nDebug - Raw research response:", content)
        
        # Parse the content based on its structure
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                # Parse the JSON string from the 'text' key
                research_data = json.loads(content[0]['text'])
            else:
                research_data = content[0]
        elif isinstance(content, dict):
            research_data = content
        else:
            research_data = json.loads(content)
            
        # Verify required keys exist
        required_keys = ['key_points', 'relevant_stats', 'target_audience', 'suggested_hashtags']
        missing_keys = [key for key in required_keys if key not in research_data]
        
        if missing_keys:
            raise KeyError(f"Missing required keys in research data: {missing_keys}")
            
        print("\nDebug - Processed research data:", research_data)
        
    except Exception as e:
        print(f"\nError processing research data: {e}")
        # Generate fallback data based on the topic
        topic_words = topic.lower().split()
        
        # Extract main keywords from topic
        keywords = [word for word in topic_words if len(word) > 3 and word not in ['and', 'the', 'for', 'with']]
        hashtags = [f"#{word.capitalize()}" for word in keywords]
        
        research_data = {
            "key_points": [
                f"Overview of {topic}",
                "Current trends and developments",
                "Key considerations and insights"
            ],
            "relevant_stats": [
                f"Recent industry data about {topic}",
                "Current market trends"
            ],
            "target_audience": "Professionals interested in " + topic,
            "suggested_hashtags": hashtags + ["#Technology", "#Innovation", "#Professional"]
        }
        print("\nUsing dynamic fallback data due to error")
    
    # Add delay between API calls
    print("\nWaiting between API calls...")
    await asyncio.sleep(1)
    
    # Step 2: Content Writing Phase
    print("\n2. Writing the initial post...")
    writing_prompt = f"""
    Create a LinkedIn post about {topic} using these insights:
    Key Points: {research_data['key_points']}
    Stats: {research_data['relevant_stats']}
    Target Audience: {research_data['target_audience']}
    """
    
    draft_response = await orchestrator.route_request(
        writing_prompt,
        user_id="linkedin_generator",
        session_id="post_gen"
    )
    draft_content = draft_response.output.content
    
    # Add delay between API calls
    print("\nWaiting between API calls...")
    await asyncio.sleep(2)
    
    # Step 3: Optimization Phase
    print("\n3. Optimizing the post...")
    optimization_prompt = f"""
    Create a professional LinkedIn post in markdown format about this topic.
    Use proper markdown syntax for formatting.
    Include these hashtags at the end: {research_data['suggested_hashtags']}
    
    Topic: {topic}
    Post to optimize:
    {draft_content}
    
    Requirements:
    1. Use ## for section headers
    2. Use bold (**) for emphasis
    3. Use proper line breaks between sections
    4. Format lists with proper markdown syntax
    5. Add hashtags at the end in a clean format
    6. Return only the formatted content, no JSON or list structures
    """
    
    final_response = await orchestrator.route_request(
        optimization_prompt,
        user_id="linkedin_generator",
        session_id="post_gen"
    )
    
    # Clean up the response content
    content = final_response.output.content
    if isinstance(content, (list, dict)):
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                content = content[0]['text']
            else:
                content = content[0]
    
    # Format the post in markdown
    markdown_post = f"""# {topic}

{content}

---
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Create openai_posts directory if it doesn't exist
    os.makedirs('openai_posts', exist_ok=True)
    
    # Create filename from topic (sanitize the topic for filename)
    filename = topic.lower().replace(' ', '_').replace(':', '').replace('/', '_')
    filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'openai_posts/{filename}_{timestamp}.md'
    
    # Save the post
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_post)
    
    print(f"\nPost saved to: {filepath}")
    return markdown_post

async def main():
    topic = input("Enter a topic: ")
    
    print(f"\n{'='*50}")
    print(f"Generating LinkedIn post for: {topic}")
    print(f"{'='*50}")
    
    final_post = await generate_linkedin_post(topic)
    print("\nFinal LinkedIn Post:")
    print(f"{'='*50}")
    print(final_post)
    print(f"{'='*50}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
