import uuid
import asyncio
import time
from typing import Dict, Optional, List
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.types import ConversationMessage
import datetime
from os_model import GeminiAgent, GeminiAgentOptions
import os
from dotenv import load_dotenv
import logging
from asyncio import sleep

load_dotenv()

# Set up logging configuration
def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent_activity.log'),
            logging.StreamHandler()  # This will print to console
        ]
    )
    # Reduce verbosity of specific loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

class GeminiResponseFormatter:
    """Helper class to format Gemini API responses"""
    @staticmethod
    def extract_content(response_text: str) -> str:
        """Extract only the relevant content from Gemini response"""
        try:
            import json
            response_dict = json.loads(response_text)
            return response_dict["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return response_text

    @staticmethod
    def log_response_metadata(response_text: str):
        """Log useful metadata from the response"""
        try:
            import json
            response_dict = json.loads(response_text)
            metadata = response_dict.get("usageMetadata", {})
            logging.info(f"Token Usage - Prompt: {metadata.get('promptTokenCount', 'N/A')}, "
                        f"Response: {metadata.get('candidatesTokenCount', 'N/A')}, "
                        f"Total: {metadata.get('totalTokenCount', 'N/A')}")
        except:
            pass

class BaseGeminiAgent(GeminiAgent):
    """Base class for all Gemini agents with improved logging"""
    async def process_request(self, *args, **kwargs) -> ConversationMessage:
        try:
            logging.info(f"Agent {self.__class__.__name__} processing request...")
            response = await super().process_request(*args, **kwargs)
            
            if hasattr(response, '_raw_response'):
                GeminiResponseFormatter.log_response_metadata(response._raw_response)
            
            logging.info(f"Agent {self.__class__.__name__} completed successfully")
            return response
        except Exception as e:
            logging.error(f"Agent {self.__class__.__name__} error: {str(e)}")
            raise

class TrendResearchAgent(BaseGeminiAgent):
    """Agent responsible for researching trending topics with statistics."""
    
    def __init__(self):
        super().__init__(GeminiAgentOptions(
            name="Trend Researcher",
            description="Researches current trends and statistics in the given domain",
            api_key=os.getenv("GEMINI_API_KEY")
        ))

    def can_handle(self, task: Dict) -> bool:
        """Let the orchestrator know what tasks this agent can handle"""
        return (
            task.get("task_type") == "research" or
            "research" in task.get("description", "").lower()
        )

    async def process_request(self, input_text: str, user_id: str, session_id: str, 
                            chat_history: List[ConversationMessage], 
                            additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage:
        prompt = f"""Research and provide EXACTLY 3 current, specific insights about: {input_text}
        Focus on latest statistics, market trends, and business impact.
        Format EXACTLY as follows:
        1. [Specific statistic/number from last 3 months] + [One-line business impact]
        2. [Specific statistic/number from last 3 months] + [One-line business impact]
        3. [Specific statistic/number from last 3 months] + [One-line business impact]"""
        
        return await super().process_request(prompt, user_id, session_id, chat_history, additional_params)

class CompetitorAnalysisAgent(BaseGeminiAgent):
    """Agent responsible for analyzing competitor strategies."""
    
    def __init__(self):
        super().__init__(GeminiAgentOptions(
            name="Competitor Analyst",
            description="Analyzes competitor strategies and market positioning",
            api_key=os.getenv("GEMINI_API_KEY")
        ))

    def can_handle(self, task: Dict) -> bool:
        return (
            task.get("task_type") == "analysis" or
            "competitor" in task.get("description", "").lower()
        )

    async def process_request(self, input_text: str, user_id: str, session_id: str, 
                            chat_history: List[ConversationMessage], 
                            additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage:
        prompt = f"""Analyze top companies' strategies in: {input_text}
        Provide 2 specific examples of successful implementations or approaches.
        Format:
        1. [Company Name]: [Specific strategy/approach] -> [Measurable outcome]
        2. [Company Name]: [Specific strategy/approach] -> [Measurable outcome]"""
        
        return await super().process_request(prompt, user_id, session_id, chat_history, additional_params)

class HashtagRecommenderAgent(BaseGeminiAgent):
    """Agent responsible for suggesting relevant hashtags."""
    
    def __init__(self):
        super().__init__(GeminiAgentOptions(
            name="Hashtag Recommender",
            description="Suggests trending and relevant hashtags",
            api_key=os.getenv("GEMINI_API_KEY")
        ))

    def can_handle(self, task: Dict) -> bool:
        return (
            task.get("task_type") == "hashtags" or
            "hashtag" in task.get("description", "").lower()
        )

    async def process_request(self, input_text: str, user_id: str, session_id: str, 
                            chat_history: List[ConversationMessage], 
                            additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage:
        prompt = f"""Suggest 5 highly relevant and trending hashtags for a LinkedIn post about: {input_text}
        Include a mix of:
        - Industry-specific tags
        - Trending technology tags
        - Professional development tags
        Format: #tag1 #tag2 #tag3 #tag4 #tag5"""
        
        return await super().process_request(prompt, user_id, session_id, chat_history, additional_params)

class LinkedInPostGenerator(BaseGeminiAgent):
    """Agent responsible for generating the final LinkedIn post."""
    
    def __init__(self):
        super().__init__(GeminiAgentOptions(
            name="LinkedIn Post Generator",
            description="Creates engaging LinkedIn posts with research",
            api_key=os.getenv("GEMINI_API_KEY")
        ))

    def can_handle(self, task: Dict) -> bool:
        return (
            task.get("task_type") == "post_generation" or
            "post" in task.get("description", "").lower()
        )

    async def process_request(self, input_text: str, user_id: str, session_id: str, 
                            chat_history: List[ConversationMessage], 
                            additional_params: Optional[Dict[str, str]] = None) -> ConversationMessage:
        context = additional_params.get('context', {}) if additional_params else {}
        research = context.get('research_results', '')
        competitor_analysis = context.get('competitor_analysis', '')
        hashtags = context.get('hashtags', '')
        
        prompt = f"""Create an engaging LinkedIn post about {input_text}.
        
        Use these insights:
        {research}
        
        Consider these competitor strategies:
        {competitor_analysis}
        
        Format:
        🚀 [Hook using the most impressive statistic]
        
        [2 lines explaining why this matters to professionals]
        
        Key Insights:
        📊 [First insight with statistic]
        📈 [Second insight with statistic]
        
        Industry Examples:
        [Mention one competitor example]
        
        💡 [One powerful conclusion about future impact]
        
        [Engaging question]
        
        {hashtags}"""
        
        return await super().process_request(prompt, user_id, session_id, chat_history, additional_params)

class LinkedInPostWorkflow:
    def __init__(self):
        # Initialize orchestrator with proper configuration
        self.orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
            LOG_AGENT_CHAT=True,
            LOG_CLASSIFIER_CHAT=True,
            LOG_CLASSIFIER_OUTPUT=True,
            LOG_EXECUTION_TIMES=True,
            MAX_RETRIES=3,
            USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
            MAX_MESSAGE_PAIRS_PER_AGENT=10
        ))
        
        # Register agents
        self.trend_researcher = TrendResearchAgent()
        self.competitor_analyst = CompetitorAnalysisAgent()
        self.hashtag_recommender = HashtagRecommenderAgent()
        self.post_generator = LinkedInPostGenerator()
        
        # Add agents to orchestrator
        self.orchestrator.add_agent(self.trend_researcher)
        self.orchestrator.add_agent(self.competitor_analyst)
        self.orchestrator.add_agent(self.hashtag_recommender)
        self.orchestrator.add_agent(self.post_generator)

    async def generate_post(self, topic: str, user_id: str, session_id: str) -> str:
        try:
            # Step 1: Initial research request
            research_prompt = f"Analyze current trends and statistics for: {topic}"
            research_result = await self.trend_researcher.process_request(
                research_prompt,
                user_id,
                session_id,
                [],
                {"task_type": "research"}
            )
            await asyncio.sleep(1)  # Add delay between requests

            # Step 2: Competitor analysis based on research
            competitor_prompt = f"Based on these trends: {research_result.content[0]['text']}, analyze competitor strategies in {topic}"
            competitor_result = await self.competitor_analyst.process_request(
                competitor_prompt,
                user_id,
                session_id,
                [],
                {"task_type": "analysis"}
            )
            await asyncio.sleep(1)

            # Step 3: Generate hashtags based on context
            hashtag_prompt = f"Suggest hashtags for {topic} considering: {research_result.content[0]['text']}"
            hashtag_result = await self.hashtag_recommender.process_request(
                hashtag_prompt,
                user_id,
                session_id,
                [],
                {"task_type": "hashtags"}
            )
            await asyncio.sleep(1)

            # Step 4: Generate final post with all context
            final_result = await self.post_generator.process_request(
                topic,
                user_id,
                session_id,
                [],
                {
                    "task_type": "post_generation",
                    "context": {
                        "research_results": research_result.content[0]["text"],
                        "competitor_analysis": competitor_result.content[0]["text"],
                        "hashtags": hashtag_result.content[0]["text"]
                    }
                }
            )

            return final_result.content[0]["text"]

        except Exception as e:
            logging.error(f"Workflow error: {str(e)}")
            raise

def save_to_markdown(content: str, filename: str) -> None:
    """Save the generated LinkedIn post to a markdown file in the posts directory."""
    try:
        # Create posts directory if it doesn't exist
        posts_dir = "generated_posts"
        os.makedirs(posts_dir, exist_ok=True)
        
        # Create full path for the file
        file_path = os.path.join(posts_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nPost saved to: {file_path}")
    except Exception as e:
        print(f"\nError saving file: {e}")

def main():
    setup_logging()
    logging.info("Starting LinkedIn Post Generator")
    
    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())
    logging.info(f"Session initialized with ID: {SESSION_ID}")
    
    print("Welcome to the LinkedIn Post Generator. Type 'quit' to exit.")
    print("Posts will be saved in the 'generated_posts' directory.")
    
    workflow = LinkedInPostWorkflow()
    
    while True:
        user_input = input("\nEnter the topic for your LinkedIn post: ").strip()
        
        if user_input.lower() == 'quit':
            logging.info("User requested to quit the application")
            print("Exiting the program. Goodbye!")
            break
        
        try:
            logging.info(f"Processing new request for topic: {user_input}")
            final_post = asyncio.run(workflow.generate_post(user_input, USER_ID, SESSION_ID))
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = user_input.lower().replace(' ', '_')[:30]
            filename = f"{topic_slug}_{timestamp}.md"
            
            if not final_post.startswith("Error:"):
                save_to_markdown(final_post, filename)
                logging.info(f"Successfully generated and saved post for topic: {user_input}")
            else:
                logging.error(f"Failed to generate post for topic: {user_input}")
            
            print("\n=== Final LinkedIn Post ===\n")
            print(final_post)
            print("\n========================\n")
            
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


