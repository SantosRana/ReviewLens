# src/groq_client.py
import os
import json
from typing import List
from groq import Groq
from dotenv import load_dotenv
from src.schema import BatchABSAResponse, ReviewInput

load_dotenv()

class GroqEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=api_key)
        
        # Unified Model for all tasks
        self.model = "openai/gpt-oss-120b"


    def analyze_reviews_batch(self, batch: List[ReviewInput]) -> BatchABSAResponse:
        # Format batch explicitly exposing ReviewInput.id to the LLM
        formatted_reviews = "\n\n".join([
            f"[REVIEW_ID={r.id}]\n{r.text}" for r in batch
        ])
        
        schema_json = json.dumps(BatchABSAResponse.model_json_schema(), indent=2)

        system_prompt = (
        "You are a Korean Restaurant Aspect-Based Sentiment Analysis (ABSA) engine.\n"
        "Analyze every provided review independently and respond strictly in valid JSON format.\n\n"
        "CRITICAL REVIEW_ID CONTRACT:\n"
        "- Each review is prefixed with [REVIEW_ID=X].\n"
        "- In your JSON output, set 'review_id' to the EXACT integer value provided in [REVIEW_ID=X].\n"
        "- NEVER change, invent, or reorder REVIEW_ID values.\n\n"
        "ALLOWED ASPECTS & SENTIMENTS:\n"
        "- 'aspect' MUST be one of: FOOD, PRICE, SERVICE, AMBIENCE\n"
        "- 'sentiment' MUST be one of: positive, negative, neutral\n\n"
        "REQUIRED FIELDS & SUMMARY CONTRACT:\n"
        "- Every review object MUST contain 'review_id', 'summary', and 'aspects'.\n"
        "- If no summary can be generated, set 'summary': \"\". Never omit the field.\n\n"
        "EVIDENCE FIDELITY RULE:\n"
        "- 'evidence' MUST be a 100% EXACT VERBATIM SUBSTRING copied directly from that specific review_id.\n"
        "- DO NOT summarize, paraphrase, fix typos, or modify punctuation.\n\n"
        "Output valid JSON matching this schema:\n"
        f"{schema_json}"
        ) 

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze these reviews and output valid JSON:\n{formatted_reviews}"} # <--- Added 'JSON' here as well
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4096
        )

        return BatchABSAResponse.model_validate_json(response.choices[0].message.content)
    
    
    def generate_chat_response(self, system_instruction: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content