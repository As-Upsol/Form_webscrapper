from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import os
import asyncio
import httpx

from crawl4ai.async_webcrawler import AsyncWebCrawler
from matching_prompts import get_analysis_prompts

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"
PROMPTS_FILE = "prompts.txt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-search-preview"

SYSTEM_PROMPT = """You are a senior M&A analyst and ex-investment banker with deep experience in commercial due diligence and buyer targeting.
Your role is to analyze text extracted from a company's website and extract clear, structured business information for use in buy-side and sell-side M&A processes.
You focus on what the company actually does, how it delivers value, and in which domain — avoiding marketing language, vague benefits, or aspirational claims.
Your output is concise, analytical, and aligned with the information needs of investors and acquirers. You prioritize function over fluff, and always return structured, match-ready data that can support semantic scoring and deal logic — with no explanation.
Do include your thought process, just give the answers with no extra explanations or opinion.
Do not output introductions, analysis, features, or commentary
Return just result — no reasoning, no summary, no preamble.
"""

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class GroqRequest(BaseModel):
    url: HttpUrl
    prompts: Optional[List[str]] = None

class AnalysisRequest(BaseModel):
    base_responses: Dict[str, str]

def load_prompts_from_file(markdown: str) -> List[Dict[str, str]]:
    if not os.path.exists(PROMPTS_FILE):
        raise RuntimeError(f"{PROMPTS_FILE} not found.")
    prompts = []
    with open(PROMPTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Split only on the first colon
            title, prompt = line.split(":", 1)
            title = title.strip().strip('"')
            prompt = prompt.strip().replace("{MARKDOWN}", markdown)
            prompts.append({"title": title, "prompt": prompt})
    return prompts

def parse_request_prompts(request_prompts: List[str], markdown: str) -> List[Dict[str, str]]:
    prompts = []
    for line in request_prompts:
        if not line or ":" not in line:
            continue
        title, prompt = line.split(":", 1)
        title = title.strip().strip('"')
        prompt = prompt.strip().replace("{MARKDOWN}", markdown)
        prompts.append({"title": title, "prompt": prompt})
    return prompts

async def query_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

async def fetch_html(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=20)
        response.raise_for_status()
        return response.text

async def openai_markdown_from_html(html: str, url: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    # Prompt OpenAI to convert HTML to markdown
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": html}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

async def query_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

@app.post("/process")
async def process_groq_request(request: GroqRequest) -> Dict[str, Any]:
    markdown = None
    html = None
    crawl4ai_error = None
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(str(request.url))
            markdown = result.markdown.raw_markdown if result.markdown else None
            html = result.html
    except Exception as e:
        crawl4ai_error = str(e)
    # Fallback if crawl4ai fails or markdown is empty
    if not markdown:
        try:
            html = await fetch_html(str(request.url))
            markdown = await openai_markdown_from_html(html, str(request.url))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Both crawl4ai and OpenAI fallback failed: {crawl4ai_error} | {str(e)}")
    # Load prompts from file or request
    prompt_objs = load_prompts_from_file(markdown)
    # Query Groq for each prompt, fallback to OpenAI if Groq fails
    groq_responses = {}
    for prompt_obj in prompt_objs:
        try:
            answer = await query_groq(prompt_obj["prompt"])
        except Exception:
            # Fallback to OpenAI for this prompt
            try:
                answer = await query_openai(prompt_obj["prompt"])
            except Exception as e:
                answer = f"Both Groq and OpenAI failed for this prompt: {str(e)}"
        groq_responses[prompt_obj["title"]] = answer
    return {"groq_responses": groq_responses}

@app.post("/process_analysis")
async def process_analysis(request: AnalysisRequest) -> Dict[str, Any]:
    groq_responses = request.base_responses

    # Step 2: Extract required fields for analysis prompts
    long_offering = groq_responses.get("Company_Offering", "")
    summary = groq_responses.get("Company_Description", "")
    long_problem_solved = groq_responses.get("Problem_Solved_Market_Pain_Point", "")
    long_use_cases = groq_responses.get("Use_Cases_and_End_Users", "")
    target_customers_description = groq_responses.get("Target_Customer_Description", "")

    # Step 3: Generate analysis prompts
    analysis_prompts_dict = get_analysis_prompts(
        long_offering, summary, long_problem_solved, long_use_cases, target_customers_description
    )
    analysis_responses = {}
    for title, prompt in analysis_prompts_dict.items():
        try:
            answer = await query_groq(prompt)
        except Exception:
            try:
                answer = await query_openai(prompt)
            except Exception as e:
                answer = f"Both Groq and OpenAI failed for this prompt: {str(e)}"
        analysis_responses[title] = answer

    return {
        "analysis_responses": analysis_responses
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 