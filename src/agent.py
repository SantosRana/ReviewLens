# src/agent.py
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from groq import Groq
from dotenv import load_dotenv

from src.absa import ABSAOrchestrator
from src.naver import NaverBlogSearch
from src.guardrails import SecurityGuardrails

logger = logging.getLogger(__name__)
load_dotenv()


class ReviewLensAgent:
    def __init__(self):
        # 1. Load Environment & API Client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"
        
        # 2. Load Core Dependencies
        self.guardrails = SecurityGuardrails(max_input_length=1000)
        self.orchestrator = ABSAOrchestrator(batch_size=7)
        self.naver_search = NaverBlogSearch()
        
        # Load Modular Skills (.md files directly under skills/)
        self.domain_guard_skill = self._load_skill("domain_guard")
        self.absa_skill = self._load_skill("bilingual_absa_response")

        # 3. Explicit State Initialization
        self.current_restaurant: Optional[str] = None
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}  
        
        # 4. Loop & Tool Limits
        self.max_iterations: int = 5

    def _load_skill(self, skill_filename: str) -> str:
        """Loads prompt guidelines directly from a <skill_filename>.md file in the skills directory."""
        skill_path = Path(__file__).resolve().parent.parent / "skills" / f"{skill_filename}.md"
        
        if skill_path.exists():
            return skill_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"[Agent] Skill file not found at: {skill_path}")
            return ""

    def _get_system_instructions(self) -> str:
        base_instruction = (
            "You are ReviewLens, an AI agent for Korean Restaurant Aspect-Based Sentiment Analysis.\n"
            "DEFAULT CONVERSATIONAL LANGUAGE: English.\n"
            "Always converse in English unless the user's prompt is written in other languages.\n"
            "You help users inspect aspect scores (FOOD, PRICE, SERVICE, AMBIENCE) and grounded review quotes.\n\n"
        )
        
        # ONLY append response formatting guidelines (bilingual_absa_response.md)
        if self.absa_skill:
            base_instruction += f"--- RESPONSE FORMATTING & LANGUAGE RULES ---\n{self.absa_skill}\n\n"
            
        return base_instruction

    def _trim_for_llm(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a lightweight dict for LLM context while preserving engineering metrics."""
        return {
            "status": analysis_result.get("status", "success"),
            "restaurant_name": analysis_result.get("restaurant_name"),
            "total_reviews_analyzed": analysis_result.get("total_reviews_analyzed", 0),
            "overall_score": analysis_result.get("overall_score", 0.0),
            "aspect_scores": analysis_result.get("aspect_scores", {}),
            "pros": analysis_result.get("pros", [])[:3],
            "cons": analysis_result.get("cons", [])[:3],
            "grounding_rate": analysis_result.get("grounding_rate", 0.0),
            "pipeline_metrics": analysis_result.get("pipeline_metrics", {})
        }

    
    def classify_domain_intent(self, user_query: str) -> str:
        """Classifies user intent into REVIEW, FOLLOW_UP, CASUAL, or OFF_TOPIC."""
        if not self.domain_guard_skill:
            logger.warning("[Agent] domain_guard.md missing — defaulting intent to CASUAL.")
            return "CASUAL"

        system_prompt = (
            "You are the domain intent classifier for ReviewLens.\n\n"
            f"{self.domain_guard_skill}\n\n"
            "Output ONLY the category name in capital letters (REVIEW, FOLLOW_UP, CASUAL, OFF_TOPIC)."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0
        )

        raw_intent = response.choices[0].message.content.strip().upper()
    
        # Clean up any quotes or markdown backticks if the LLM outputted them
        intent = raw_intent.replace("`", "").replace('"', '').replace("'", "")
        valid_intents = {"REVIEW", "FOLLOW_UP", "CASUAL", "OFF_TOPIC"}

        return intent if intent in valid_intents else "CASUAL"
    
    # ------------------------------------------------------------------
    # Tool Implementations with State Caching
    # ------------------------------------------------------------------
    def search_and_analyze_restaurant(self, restaurant_name: str) -> Dict[str, Any]:
        """Tool: Fetches Naver blog reviews for a restaurant and performs ABSA analysis."""
        if restaurant_name in self.analysis_cache:
            logger.info(f"Using cached ABSA state for '{restaurant_name}'")
            self.current_restaurant = restaurant_name
            return self._trim_for_llm(self.analysis_cache[restaurant_name])

        try:
            reviews = self.naver_search.fetch_reviews(restaurant_name, display_count=20)
            if not reviews:
                return {"status": "error", "message": f"No reviews found for restaurant: '{restaurant_name}'."}

            absa_results = self.orchestrator.process_reviews(reviews)
            absa_results["restaurant_name"] = restaurant_name
            absa_results["status"] = "success"

            self.analysis_cache[restaurant_name] = absa_results
            self.current_restaurant = restaurant_name
            self.current_analysis = absa_results
            return self._trim_for_llm(absa_results)

        except Exception as e:
            logger.error(f"Error in search_and_analyze_restaurant: {e}")
            return {"status": "error", "message": str(e)}

    def compare_restaurants(self, restaurant_a: str, restaurant_b: str) -> Dict[str, Any]:
        """Fetches and analyzes reviews for two restaurants side-by-side."""
        res_a = self.search_and_analyze_restaurant(restaurant_a)
        res_b = self.search_and_analyze_restaurant(restaurant_b)

        winner = "Tie"
        if res_a.get("status") == "success" and res_b.get("status") == "success":
            score_a = res_a.get("overall_score", 0.0)
            score_b = res_b.get("overall_score", 0.0)
            if score_a > score_b:
                winner = restaurant_a
            elif score_b > score_a:
                winner = restaurant_b

        return {
            "status": "success",
            "winner": winner,
            "restaurant_a": res_a,
            "restaurant_b": res_b
        }

    def get_aspect_insights(self, aspect: str, restaurant_name: Optional[str] = None) -> Dict[str, Any]:
        """Tool: Extracts evidence, sentiment scores, and quotes for a specific aspect from current analysis."""
        target = restaurant_name or self.current_restaurant
        if not target or target not in self.analysis_cache:
            return {
                "status": "error",
                "message": "No active restaurant analysis found. Search for a restaurant first."
            }

        analysis = self.analysis_cache[target]
        aspect_upper = aspect.upper()
        aspect_scores = analysis.get("aspect_scores", {}).get(aspect_upper, {})

        matching_evidence = []
        for review in analysis.get("details", []):
            for item in review.aspects:
                if item.aspect == aspect_upper:
                    matching_evidence.append({
                        "sentiment": item.sentiment,
                        "evidence": item.evidence,
                        "summary": review.summary
                    })

        return {
            "status": "success",
            "restaurant_name": target,
            "aspect": aspect_upper,
            "aspect_metrics": aspect_scores,
            "total_mentions": len(matching_evidence),
            "evidence_list": matching_evidence[:8]
        }

    def analyze_custom_reviews(self, reviews: List[str]) -> Dict[str, Any]:
        """Tool: Sanitizes and analyzes custom review text directly provided in user prompt."""
        if not reviews:
            return {"status": "error", "message": "No reviews provided."}

        sanitized_reviews: List[str] = []
        rejected_count = 0

        for review in reviews:
            if not review or not review.strip():
                continue

            sanitized = self.guardrails.validate_review_data(review)
            if sanitized is not None:
                sanitized_reviews.append(sanitized)
            else:
                rejected_count += 1

        if rejected_count > 0:
            logger.warning(
                "[Guardrails] Rejected %d unsafe custom review entries", rejected_count
            )

        if not sanitized_reviews:
            return {
                "status": "error",
                "message": (
                    "All provided review text was empty or "
                    "rejected by security guardrails."
                ),
            }

        results = self.orchestrator.process_reviews(sanitized_reviews)
        results["status"] = "success"

        self.current_restaurant = "Custom Input"
        self.current_analysis = results
        return self._trim_for_llm(results)

    # ------------------------------------------------------------------
    # Tool Schemas for Groq Function Calling
    # ------------------------------------------------------------------
    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_and_analyze_restaurant",
                    "description": "Fetch Naver blog reviews for a Korean restaurant and perform Aspect-Based Sentiment Analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant_name": {
                                "type": "string",
                                "description": "Name of the restaurant (e.g., '봉피양 강남점', '삼청동수제비')."
                            }
                        },
                        "required": ["restaurant_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_aspect_insights",
                    "description": "Extract granular evidence, sentiment scores, and quotes for a specific aspect (FOOD, PRICE, SERVICE, or AMBIENCE) from a previously analyzed restaurant.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "aspect": {
                                "type": "string",
                                "enum": ["FOOD", "PRICE", "SERVICE", "AMBIENCE"],
                                "description": "The specific aspect category to inspect."
                            },
                            "restaurant_name": {
                                "type": "string",
                                "description": "Optional — which analyzed restaurant to inspect. Defaults to the most recently analyzed one if omitted."
                            }
                        },
                        "required": ["aspect"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_restaurants",
                    "description": "Fetch reviews and compare two Korean restaurants side-by-side across all aspects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "restaurant_a": {"type": "string", "description": "Name of first restaurant."},
                            "restaurant_b": {"type": "string", "description": "Name of second restaurant."}
                        },
                        "required": ["restaurant_a", "restaurant_b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_custom_reviews",
                    "description": "Analyze raw review text strings directly provided by the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reviews": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of raw review texts to analyze."
                            }
                        },
                        "required": ["reviews"]
                    }
                }
            }
        ]

    # ------------------------------------------------------------------
    # Agent Tool Loop
    # ------------------------------------------------------------------
    def run(self, user_query: str) -> Dict[str, Any]:
        """Executes security guardrails, handles tool invocation, and generates conversational output."""
       # 1. Security Check (Boundary 1)
        guardrail_result = self.guardrails.validate_user_input(user_query)
        if not guardrail_result.is_safe:
            return {
                "response_text": guardrail_result.error_message,
                "tool_data": None,
                "active_restaurant": self.current_restaurant
            }

        # 2. Domain Intent Routing
        intent = self.classify_domain_intent(user_query)

        if intent in {"CASUAL", "OFF_TOPIC"}:
            if intent == "CASUAL":
                response_msg = (
                    "Hello! I am ReviewLens, your AI assistant for Korean restaurant review intelligence. "
                    "I can analyze Korean restaurant reviews, compare dining spots, and provide "
                    "evidence-grounded aspect scores for Food, Price, Service, and Ambience. "
                    "How can I help you today?"
                )
            else:
                response_msg = (
                    "I am ReviewLens, an assistant focused specifically on Korean restaurant review intelligence. "
                    "I can't help with general knowledge or off-topic queries, but I'd be happy to analyze "
                    "a Korean restaurant or compare dining spots for you!"
                )

            return {
                "response_text": response_msg,
                "tool_data": None,
                "active_restaurant": self.current_restaurant
            }

        # 3. Proceed to Main Agent Tool Loop for REVIEW and FOLLOW_UP
        active_context = f" (Active Restaurant Context: {self.current_restaurant})" if self.current_restaurant else ""
        
        system_instruction = (
            self._get_system_instructions() + "\n\n"
            f"Current Session Context: {active_context}\n\n"
            "GUIDELINES:\n"
            "1. When asked about a specific aspect of an analyzed restaurant (e.g., 'How is the service?'), "
            "use `get_aspect_insights` to read from current state.\n"
            "2. When asked to search or evaluate a new restaurant, use `search_and_analyze_restaurant`.\n"
            "3. When asked to compare two restaurants, use `compare_restaurants`.\n"
            "4. Provide conversational, evidence-grounded answers citing Korean quotes when explaining reasons."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query}
        ]

        tools = self._get_tools_schema()
        last_tool_data = None

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )

            response_message = response.choices[0].message
            messages.append(response_message)

            tool_calls = getattr(response_message, "tool_calls", None)
            if not tool_calls:
                return {
                    "response_text": response_message.content,
                    "tool_data": last_tool_data or self.current_analysis,
                    "active_restaurant": self.current_restaurant
                }

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_args = {}

                tool_result = None
                if function_name == "search_and_analyze_restaurant":
                    tool_result = self.search_and_analyze_restaurant(**function_args)
                elif function_name == "get_aspect_insights":
                    tool_result = self.get_aspect_insights(**function_args)
                elif function_name == "compare_restaurants":
                    tool_result = self.compare_restaurants(**function_args)
                elif function_name == "analyze_custom_reviews":
                    tool_result = self.analyze_custom_reviews(**function_args)
                else:
                    tool_result = {"status": "error", "message": f"Unknown tool: {function_name}"}

                last_tool_data = tool_result

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

        return {
            "response_text": "I reached the maximum iteration limit while answering your request.",
            "tool_data": last_tool_data or self.current_analysis,
            "active_restaurant": self.current_restaurant
        }