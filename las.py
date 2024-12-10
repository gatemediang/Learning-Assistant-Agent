# 1. IMPORTS AND SETUP
from crewai import Agent, Task, Crew, LLM
from crewai_tools import LlamaIndexTool
from llama_index.llms.gemini import Gemini
from llama_index.readers.file import PagedCSVReader
from llama_index.core.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_parse import LlamaParse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool

from constants import *

# 2. DOCUMENT PROCESSING SETUP
# Initialize document parser for PDF files
parser = LlamaParse(api_key=parse_key, result_type="markdown")
extractor = {".pdf": parser}
# 3. LLM CONFIGURATION
# Configure Gemini as the default language model
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=API_KEY)

Settings.embed_model = GeminiEmbedding(model="models/gemini-1.5-flash", api_key=API_KEY)

# 4. DOCUMENT INDEXING
# Load and index documents from the 'docs' directory for quick retrieval
docs = SimpleDirectoryReader("docs", file_extractor=extractor).load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

# 5. TOOL SETUP
# Initialize the main LLM for the crew
crew_llm = LLM(api_key=API_KEY, model="gemini/gemini-1.5-flash")

# Create tool for accessing course materials
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Bincom GenAI Instruction Guide Lookup Tool",
    description="This tool facilitates the retrieval of specific contact sessions within the introduction to generative artificial intelligence class",
)

# 6. SEARCH TOOLS SETUP
# Initialize DuckDuckGo search with error handling
try:
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="Web Search",
        description="Search the internet for current information on a topic. Input should be a simple search string.",
        func=lambda x: search.run(str(x)),
    )
except ImportError:
    # Auto-install if package is missing
    print("Installing required package: duckduckgo-search...")
    import subprocess

    subprocess.check_call(["pip", "install", "-U", "duckduckgo-search"])
    # Retry setup after installation
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="Web Search",
        description="Search the internet for current information on a topic. Input should be a simple search string.",
        func=lambda x: search.run(str(x)),
    )
# Setup Wikipedia search tool
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wiki_tool = Tool(
    name="Wikipedia Search",
    description="Search Wikipedia for detailed information on a topic. Input should be a simple search string.",
    func=lambda x: wiki.run(str(x)),
)
# 7. AGENT SETUP
# Create the main study agent with all available tools

study_agent = Agent(
    role="Learning Assistant",
    goal=f"""Using the Bincom GenAI Instruction Guide Lookup Tool (send your query as a string), create a tailored learning guide for all topics in the instruction guide, considering a learning time of: {user_details['learning_time' ]} and preferred Langauge: {user_details['programming_language']}. Ensure the guide is engaging, easy to follow, and includes interactive elements and resources for deeper understanding.
    Create comprehensive study materials by combining instruction guide content with relevant external resources""",
    backstory="You are an expert educational content creator and researcher. You excel at finding relevant supplementary materials, creating summaries, and designing effective assessments.You are designed to empower students to achieve their goals within their chosen timeframe and preferred programming language.",
    verbose=True,
    llm=crew_llm,
    tools=[query_tool, search_tool, wiki_tool],
)

# 8. TASK DEFINITIONS
# Define various tasks for the agent (summarization, quiz creation, Q&A)
learning_guide_task = Task(
    description="""Establish a comprehensive learning path for the student, including milestones, resources, and suggested activities to enhance engagement.""",
    agent=study_agent,
    expected_output="Provide a detailed dictionary that outlines a step-by-step learning path with clear milestones and resources.",
)

summarize_task = Task(
    description="""
    For each topic in the instruction guide:
    1. First use the query tool to access the core content
    2. Then use search tools to find relevant supplementary materials:
       - Look for real-world examples and applications
       - Find recent developments or case studies
       - Identify helpful tutorials or additional resources
    3. Create a comprehensive summary that combines both internal and external content
    4. Include a 'Further Reading' section with relevant links
    5. Format everything in a clear, structured manner
    6.Find supplementary resources:
      -Use search tools to find 2-3 relevant articles or tutorials for each main topic
      -Briefly explain why each resource is helpful""",
    agent=study_agent,
    expected_output="""A dictionary containing for each topic:
    - Core concept summary
    - Real-world applications
    - Supplementary resources
    - Further reading links
    -A curated list of resources with explanations""",
)


quiz_task = Task(
    description="""
    For each topic:
    1. Create 5 quiz questions incorporating both guide content and external knowledge
    2. Include:
       - Multiple choice questions
       - Real-world scenario-based questions
       - Questions that connect concepts to current trends
    3. Provide detailed explanations with references
    """,
    agent=study_agent,
    expected_output="A dictionary of comprehensive quizzes with explanations and references",
)

qa_task = Task(
    description="""
    Act as a knowledgeable course instructor and:
    1. Use the query tool to access the course content
    2. Use search and Wikipedia tools to gather additional context when needed
    3. Answer questions about course topics clearly and accurately
    4. Provide examples and explanations that relate to real-world applications
    5. Include references to both course materials and external sources
    
    Sample questions to address:
    - What are the key concepts of this topic?
    - How is this applied in real-world scenarios?
    - What are common challenges when implementing this?
    - How does this relate to other topics in the course?
    - What is Generative AI?
    - What is the difference between Generative AI and other AI technologies?
    - What are the main components of Generative AI?
    - How does Generative AI work?
    - What are the main types of Generative AI?
    - What are the main applications of Generative AI?
    - What are the main benefits of Generative AI?
    - What are the main challenges of Generative AI?
    - What is RAG?
    - What is the difference between RAG and other AI technologies?
    - What are the main components of RAG?
    - How does RAG work?
    - What are the main types of RAG?
    - What are the main applications of RAG?
    - What are the main benefits of RAG?
    - What are the main challenges of RAG?
    -What is LLM?
    - What is the difference between LLM and other AI technologies?
    - What are the main components of LLM?
    - How does LLM work?
    - What are the main types of LLM?
    - What are the main applications of LLM?
    - What are the main benefits of LLM?
    - What are the main challenges of LLM?
    -What are transformer models?
    -What are Vertical-Specific LLMs?
    -What are the main applications of Vertical-Specific LLMs?
    -What are the main benefits of Vertical-Specific LLMs?
    -What are the main challenges of Vertical-Specific LLMs?
    -What is Prompt Engineering?
    -What are the main components of Prompt Engineering?
    -How does Prompt Engineering work?
    -What are the main types of Prompt Engineering?
    -What are the main applications of Prompt Engineering?
    -What are the main benefits of Prompt Engineering?
    -What are the main challenges of Prompt Engineering?
    """,
    agent=study_agent,
    expected_output="Detailed, well-structured answers that combine course content with external knowledge",
)


# 9. CREW SETUP AND EXECUTION
# Create and run the crew with error handling


crew = Crew(
    agents=[study_agent],  # Simplified to one agent
    tasks=[learning_guide_task, summarize_task, quiz_task, qa_task],
    verbose=True,
    task_timeout=300,  # Add timeout of 5 minutes per task
)
# 10. ERROR HANDLING AND TASK EXECUTION
try:
    result = crew.kickoff()
    print("Success! Results:")
    print(result)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Trying to proceed with individual tasks...")

    # Execute each task individually if crew execution fails
    try:
        learning_guide = learning_guide_task.execute_sync()
        print("Learning guide completed:", learning_guide)
    except Exception as e:
        print(f"Learning guide task failed: {str(e)}")

    try:
        summary = summarize_task.execute_sync()
        print("Summary completed:", summary)
    except Exception as e:
        print(f"Summary task failed: {str(e)}")

    try:
        quiz = quiz_task.execute_sync()
        print("Quiz completed:", quiz)
    except Exception as e:
        print(f"Quiz task failed: {str(e)}")

    try:
        qa = qa_task.execute_sync()
        print("Q&A completed:", qa)
    except Exception as e:
        print(f"Q&A task failed: {str(e)}")


# 11. QUERY INTERFACE
# Function to allow interactive querying of the learning assistant
# Example usage
def query_learning_assistant(question):
    try:
        # Create a temporary task for querying
        query_task = Task(
            description=f"Answer the following question: {question}",
            agent=study_agent,
            expected_output="A detailed answer to the question",
        )

        # Execute the task synchronously
        answer = query_task.execute_sync()
        print("Answer:", answer)
    except Exception as e:
        print(f"Failed to get an answer: {str(e)}")


# Example usage of the query interface

query_learning_assistant("What is Generative AI?")
query_learning_assistant("How does Generative AI differ from other AI technologies?")
