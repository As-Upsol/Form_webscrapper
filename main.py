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
GROQ_MODEL = "llama-3.1-8b-instant"
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

# async def openai_markdown_from_html(html: str, url: str) -> str:
#     if not OPENAI_API_KEY:
#         raise RuntimeError("OPENAI_API_KEY environment variable not set.")
#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     # Prompt OpenAI to convert HTML to markdown
#     data = {
#         "model": OPENAI_MODEL,
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": html}
#         ]
#     }
#     async with httpx.AsyncClient() as client:
#         response = await client.post(OPENAI_API_URL, headers=headers, json=data)
#         response.raise_for_status()
#         result = response.json()
#         return result["choices"][0]["message"]["content"]

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

async def query_groq_with_retry(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff: 2, 4, 8 seconds
                await asyncio.sleep(2 ** attempt)
            
            return await query_groq(prompt)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            continue

async def query_openai_with_retry(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff: 2, 4, 8 seconds
                await asyncio.sleep(2 ** attempt)
            
            return await query_openai(prompt)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            continue

def clean_response(text: str) -> str:
    # Remove markdown formatting
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("#", "")
    
    # Clean up bullet points
    text = text.replace("- ", "")
    text = text.replace("• ", "")
    
    # Remove extra whitespace and newlines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "; ".join(lines)
    
    # Remove extra semicolons
    text = text.replace(";;", ";")
    text = text.replace("; ;", ";")
    
    # Clean up any remaining formatting
    text = text.replace(":", ": ")
    text = text.replace("  ", " ")
    
    return text.strip()

@app.post("/process")
async def process_groq_request(request: GroqRequest) -> Dict[str, Any]:
    # Step 1: Scrape the page using crawl4ai
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(str(request.url))
            markdown = result.markdown.raw_markdown if result.markdown else None
            html = result.html
            
            if not markdown and not html:
                raise ValueError("Failed to extract content from webpage")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape webpage: {str(e)}")

    # Load prompts from file
    prompt_objs = load_prompts_from_file(markdown)
    
    # Step 2: Process each prompt through Groq with validation
    groq_responses = {}
    for prompt_obj in prompt_objs:
        title = prompt_obj["title"]
        prompt = prompt_obj["prompt"]
        
        # Add delay between requests to avoid rate limiting
        await asyncio.sleep(1)
        
        # First attempt: Query Groq
        try:
            initial_answer = await query_groq_with_retry(prompt)
            initial_answer = clean_response(initial_answer)
            
            # Add delay before validation
            await asyncio.sleep(1)
            
            # Step 3: Validate the response with a second Groq call
            validation_prompt = f"""Review this answer for accuracy and completeness. If it's incorrect or incomplete, explain why and provide the correct answer. If it's correct, just return 'VALID'.

Original prompt: {prompt}
Answer to validate: {initial_answer}"""
            
            validation_result = await query_groq_with_retry(validation_prompt)
            
            if validation_result.strip().upper() == "VALID":
                groq_responses[title] = initial_answer
            else:
                # If validation failed, try one more time with Groq
                await asyncio.sleep(1)  # Add delay before retry
                try:
                    second_attempt = await query_groq_with_retry(prompt)
                    groq_responses[title] = clean_response(second_attempt)
                except Exception:
                    raise Exception("Groq second attempt failed")
                    
        except Exception as groq_error:
            # Step 4: Fallback to OpenAI if Groq fails
            await asyncio.sleep(2)  # Longer delay before switching to OpenAI
            try:
                openai_prompt = f"""Analyze this URL and answer based on this prompt: {prompt}
                URL: {request.url}"""
                answer = await query_openai_with_retry(openai_prompt)
                groq_responses[title] = clean_response(answer)
            except Exception as e:
                groq_responses[title] = f"Failed to get response after retries: {str(e)}"

    return {"groq_responses": groq_responses}

@app.post("/process_analysis")
async def process_analysis(request: AnalysisRequest) -> Dict[str, Any]:
    groq_responses = request.base_responses

    # Step 2: Extract required fields for analysis prompts
    long_offering = groq_responses.get("long_offering", "")
    summary = groq_responses.get("summary", "")
    long_problem_solved = groq_responses.get("long_problem_solved", "")
    long_use_cases = groq_responses.get("long_use_cases", "")
    target_customers_description = groq_responses.get("target_customers_description", "")

    industry_category= groq_responses.get('industry_category','')
    product_service_tags= groq_responses.get('product_service_tags','')
    technology_delivery= groq_responses.get('technology_delivery','')
    supply_chain_role= groq_responses.get('supply_chain_role','')
    target_functional_category= groq_responses.get('target_functional_category','')
    target_customer_type= groq_responses.get('target_customer_type','')
    target_customer_industries= groq_responses.get('target_customer_industries','')



    # Step 3: Generate analysis prompts
    analysis_prompts_dict = get_analysis_prompts(
        long_offering, summary, long_problem_solved, long_use_cases, target_customers_description
    )
    analysis_responses = {
        'industry_category': industry_category,
        'product_service_tags': product_service_tags,
        'technology_delivery': technology_delivery,
        'supply_chain_role': supply_chain_role,
        'target_functional_category': target_functional_category,
        'target_customer_type': target_customer_type,
        'target_customer_industries': target_customer_industries,
        'long_offering': long_offering,
        'summary': summary,
        'long_problem_solved': long_problem_solved,
        'long_use_cases': long_use_cases,
        'target_customers_description': target_customers_description,
    }

    for title, prompt in analysis_prompts_dict.items():
        try:
            answer = await query_groq_with_retry(prompt)
            analysis_responses[title] = clean_response(answer)
        except Exception:
            try:
                answer = await query_openai_with_retry(prompt)
                analysis_responses[title] = clean_response(answer)
            except Exception as e:
                analysis_responses[title] = f"Failed to get response after retries: {str(e)}"

    return {
        "analysis_responses": analysis_responses
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 