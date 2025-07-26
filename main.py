# main.py
# --- Imports ---
import os
import asyncio
import yaml
import pypdf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.ai.generativelanguage as glm
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from typing import List, Dict, Any
from logger import logger # Import the logger instance

# --- Configuration & Initialization ---
load_dotenv()

# --- ACTION REQUIRED: Set your name and profiles here ---
YOUR_NAME = "Anubhav Gupta" # <-- Replace with your name
PROFILE_URLS = [
    "https://www.linkedin.com/in/anubhavgupta14/",
    "https://github.com/Anubhavgupta14"
]

# Configure APIs
gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please create a .env file.")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found. Please create a .env file.")

genai.configure(api_key=gemini_api_key)
tavily_client = TavilyClient(api_key=tavily_api_key)

# Initialize FastAPI app
app = FastAPI()

# --- Prompt and Knowledge Base Loading ---

def load_prompts(file_path: str = "prompt.yaml") -> Dict[str, Any]:
    """Loads prompts from a YAML file."""
    try:
        with open(file_path, 'r') as f:
            prompts = yaml.safe_load(f)
        logger.info("Prompts loaded successfully from prompt.yaml")
        return prompts
    except FileNotFoundError:
        logger.error(f"{file_path} not found. Please create it.")
        raise
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        raise

PROMPTS = load_prompts()

def read_pdf_content(file_path: str = "resume.pdf") -> str:
    """Extracts text from a PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        logger.info(f"Successfully extracted text from {file_path}")
        return text
    except FileNotFoundError:
        logger.warning(f"'{file_path}' not found. Resume content will be missing.")
        return f"Could not find the resume file at '{file_path}'."
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return f"Error processing PDF file: {e}"

def scrape_url_content(url: str) -> str:
    """Fetches and extracts text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        logger.info(f"Successfully scraped content from {url}")
        return text
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        return f"Could not retrieve content from {url}."

def load_knowledge_base() -> str:
    """Loads and combines all personalized information."""
    logger.info("Loading knowledge base...")
    resume_text = read_pdf_content("resume.pdf")
    scraped_content = [f"Content from {url}:\n{scrape_url_content(url)}" for url in PROFILE_URLS]
    full_context = "\n\n---\n\n".join([resume_text] + scraped_content)
    logger.info("Knowledge base loaded successfully.")
    return full_context

# --- AI Model, Tools, and Chat Logic ---

def tavily_search(query: str) -> str:
    """Performs a web search using Tavily."""
    logger.info(f"Performing Tavily search for: '{query}'")
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        result_texts = [f"Source: {res['url']}\nContent: {res['content']}" for res in response.get('results', [])]
        return "\n\n".join(result_texts)
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Error performing search for '{query}'."

# Define the Tavily tool for the Gemini model
tavily_tool = glm.Tool(
    function_declarations=[
        glm.FunctionDeclaration(
            name='tavily_search',
            description=PROMPTS['tavily_tool_description'].format(your_name=YOUR_NAME),
            parameters=glm.Schema(type=glm.Type.OBJECT, properties={'query': glm.Schema(type=glm.Type.STRING)})
        )
    ]
)

KNOWLEDGE_BASE = load_knowledge_base()
SYSTEM_PROMPT = PROMPTS['system_prompt'].format(your_name=YOUR_NAME, knowledge_base=KNOWLEDGE_BASE)
INITIAL_GREETING = PROMPTS['initial_greeting'].format(your_name=YOUR_NAME)

model = genai.GenerativeModel(
    model_name='gemini-2.5-pro',
    tools=[tavily_tool],
    system_instruction=SYSTEM_PROMPT,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client #{client_id} connected.")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client #{client_id} disconnected.")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

# --- API Endpoints ---

@app.get("/")
async def get():
    return HTMLResponse("<h1>Personal AI Assistant Backend is Running</h1>")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    # Each connection gets a new chat session, which maintains its own history.
    chat_session = model.start_chat(history=[])
    
    try:
        # Send the initial greeting
        await manager.send_personal_message(INITIAL_GREETING, client_id)

        while True:
            user_message = await websocket.receive_text()
            logger.info(f"Received message from {client_id}: {user_message}")

            try:
                # The chat session automatically handles history.
                response = await asyncio.to_thread(chat_session.send_message, user_message)
                
                # Handle tool calls if the model requests them
                if response.candidates[0].content.parts[0].function_call:
                    fc = response.candidates[0].content.parts[0].function_call
                    if fc.name == "tavily_search":
                        query = fc.args.get('query', '')
                        tool_response = tavily_search(query=query)
                        
                        # Send the tool's response back to the model
                        response = await asyncio.to_thread(
                            chat_session.send_message,
                            glm.Part(function_response=glm.FunctionResponse(name='tavily_search', response={'result': tool_response}))
                        )

                ai_response = response.text
                logger.info(f"Sending AI response to {client_id}: {ai_response}")
                await manager.send_personal_message(ai_response, client_id)

            except Exception as e:
                logger.error(f"Error during chat session for {client_id}: {e}", exc_info=True)
                error_message = "I'm sorry, an error occurred while processing your request. Please try again."
                await manager.send_personal_message(error_message, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"An unexpected error occurred with client {client_id}: {e}", exc_info=True)
        manager.disconnect(client_id)