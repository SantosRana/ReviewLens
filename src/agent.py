import json
import re
from datetime import datetime

import ollama

from .absa import analyze_reviews
from .naver import search_restaurant_reviews
from config import OLLAMA_MODEL

LANGUAGE_POLICY = (
    "Respond in English for all explanations, summaries, labels, and comparisons. "
    "Korean is allowed only for restaurant names, branch names, or short quoted review text. "
    "Do not use Chinese, Japanese, or any other non-English language in the answer body."
)

SYSTEM_PROMPT = f"""You are ReviewLens, a helpful AI assistant that specializes in Korean restaurant review analysis.
You have two modes:
1. RESTAURANT MODE: When users ask about specific restaurants, reviews, or comparisons, use your tools to fetch and analyze real data from Naver Blog.
2. CHAT MODE: For general conversation, questions, or casual chat, respond naturally without using tools.

You can analyze reviews by aspect: FOOD, PRICE, SERVICE, AMBIENCE.
Each aspect returns: positive, negative, or not_mentioned.
{LANGUAGE_POLICY}
Always write Korean restaurant names exactly as the user typed them.
Never make up reviews; always use tools to get real data when analyzing restaurants."""

DISALLOWED_SCRIPT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f"
    r"\u0400-\u04ff"
    r"\u0370-\u03ff"
    r"\u0590-\u05ff"
    r"\u0600-\u06ff"
    r"\u0900-\u097f"
    r"\u0e00-\u0e7f]"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_reviews",
            "description": "Analyze sentiment of one or multiple Korean restaurant reviews by aspect. Use when the user provides review text directly (not a restaurant name).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reviews": {
                        "type": "string",
                        "description": "A single review string or a JSON array of review strings."
                    }
                },
                "required": ["reviews"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurant_sentiment",
            "description": "Fetch Naver blog reviews for a Korean restaurant and analyze sentiment by aspect. Use when user mentions a restaurant name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_name": {
                        "type": "string",
                        "description": "Restaurant name to analyze."
                    }
                },
                "required": ["restaurant_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_restaurants",
            "description": "Compare two restaurants side by side when the user asks for a restaurant comparison.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_a": {
                        "type": "string",
                        "description": "First restaurant name."
                    },
                    "restaurant_b": {
                        "type": "string",
                        "description": "Second restaurant name."
                    }
                },
                "required": ["restaurant_a", "restaurant_b"]
            }
        }
    }
]


def has_disallowed_language(text: str) -> bool:
    """Allow English plus Korean, but block other scripts in final answers."""
    return bool(DISALLOWED_SCRIPT_RE.search(text))


def enforce_language_policy(answer: str) -> str:
    """
    Rewrite responses that drift into disallowed scripts.
    Korean names can remain, but the explanation itself must stay in English.
    """
    if not has_disallowed_language(answer):
        return answer

    rewrite_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Rewrite this answer in English. Preserve Korean restaurant names exactly as written. "
                "Korean is allowed only for restaurant names, branch names, or very short quoted review text. "
                "Do not use Chinese, Japanese, or any other non-English language in the rewritten answer.\n\n"
                f"{answer}"
            ),
        },
    ]

    for _ in range(2):
        rewrite = ollama.chat(
            model=OLLAMA_MODEL,
            messages=rewrite_messages,
        )
        candidate = rewrite["message"]["content"]
        if not has_disallowed_language(candidate):
            return candidate

        rewrite_messages.append({"role": "assistant", "content": candidate})
        rewrite_messages.append({
            "role": "user",
            "content": (
                "Try again. The answer must stay in English. "
                "Korean is allowed only for restaurant names or short quoted review text."
            ),
        })

    return (
        "I analyzed the reviews. The structured results are available in the results panel."
    )


def get_restaurant_sentiment(
    kc_model, 
    restaurant_name: str,
    progress_callback=None,
    result_callback=None
) -> dict:
    """
    Retrieve and analyze sentiment from restaurant reviews.
    """
    reviews = search_restaurant_reviews(restaurant_name, display=20)
    
    if not reviews:
        return {"error": f"No reviews found for {restaurant_name}"}

    texts = [r["description"] for r in reviews]
    
    summary = analyze_reviews(kc_model, texts, progress_callback=progress_callback)
    
    summary["restaurant"] = restaurant_name
    summary["total_reviews"] = len(texts)
    summary["reviews"] = texts
    
    if result_callback:
        result_callback(summary)
    
    return summary


def compare_restaurants(
    kc_model, 
    restaurant_a: str, 
    restaurant_b: str,
    progress_callback=None,
    result_callback=None
) -> list:
    """Compare two restaurants and return results as a list."""
    result_a = get_restaurant_sentiment(
        kc_model, 
        restaurant_a, 
        progress_callback=progress_callback,
        result_callback=result_callback
    )
    result_b = get_restaurant_sentiment(
        kc_model, 
        restaurant_b,
        progress_callback=progress_callback,
        result_callback=result_callback
    )
    return [result_a, result_b]


def run_tool(tool_name, tool_args, kc_model, progress_callback=None, result_callback=None):
    """
    Run a specified tool with given arguments and model.
    """
    
    # Handle sentiment analysis on reviews (direct text input)
    if tool_name == "analyze_reviews":
        reviews = tool_args["reviews"]
        try:
            parsed = json.loads(reviews)
        except Exception:
            parsed = reviews
        
        # Ensure it's a list
        if isinstance(parsed, str):
            parsed = [parsed]
        
        # Run analysis
        result = analyze_reviews(kc_model, parsed, progress_callback=progress_callback)
        
        # Mark as review analysis
        result["is_review_analysis"] = True
        result["review_count"] = len(parsed)
        
        # Store individual review results for display
        # Since analyze_reviews aggregates, we create individual entries
        result["reviews_data"] = []
        for review_text in parsed:
            # For each review, we store the text and the aggregated aspect data
            # Note: For true per-review aspects, you'd need to analyze each separately
            review_entry = {
                "text": review_text,
                "FOOD": result.get("FOOD", {"positive": 0, "negative": 0}),
                "PRICE": result.get("PRICE", {"positive": 0, "negative": 0}),
                "SERVICE": result.get("SERVICE", {"positive": 0, "negative": 0}),
                "AMBIENCE": result.get("AMBIENCE", {"positive": 0, "negative": 0}),
            }
            result["reviews_data"].append(review_entry)
        
        # IMPORTANT: Call result_callback for review analysis too!
        if result_callback:
            result_callback(result)
        
        return json.dumps(result, ensure_ascii=False)

    # Handle restaurant sentiment summary
    if tool_name == "get_restaurant_sentiment":
        result = get_restaurant_sentiment(
            kc_model, 
            tool_args["restaurant_name"],
            progress_callback=progress_callback,
            result_callback=result_callback
        )
        return json.dumps(result, ensure_ascii=False)

    # Handle restaurant comparison
    if tool_name == "compare_restaurants":
        result = compare_restaurants(
            kc_model,
            tool_args["restaurant_a"],
            tool_args["restaurant_b"],
            progress_callback=progress_callback,
            result_callback=result_callback
        )
        return json.dumps(result, ensure_ascii=False)

    return "Tool not found"


def is_restaurant_query(message: str) -> bool:
    """
    Detect if the user is asking for restaurant analysis.
    """
    message_lower = message.lower()

    # If it looks like a review (long text with sentiment words), treat as review
    review_indicators = ["맛있", "맛없", "좋았", "나빴", "비쌌", "쌌", "친절", "불친절", "깨끗", "더러"]
    if any(indicator in message for indicator in review_indicators) and len(message) > 20:
        return True

    strong_indicators = [
        "analyze",
        "compare",
        "vs",
        "review",
        "restaurant",
        "cafe",
        "branch",
        "분석",
        "비교",
        "차이",
        "리뷰",
        "맛집",
        "식당",
        "음식점",
        "카페",
        "지점",
    ]

    restaurant_pattern = (
        r"[\uac00-\ud7a3]{2,6}"
        r"(?:점|집|식당|카페|음식점|"
        r"치킨|피자|버거|국|탕|면)"
    )
    comparison_pattern = (
        r"(?:랑|와|과|and|or)\s+\S+?\s*"
        r"(?:비교|vs|compare|차이|대비)"
    )

    has_strong_indicator = any(indicator in message_lower for indicator in strong_indicators)
    has_restaurant_name = bool(re.search(restaurant_pattern, message))
    is_comparison = bool(re.search(comparison_pattern, message_lower))

    return has_strong_indicator or has_restaurant_name or is_comparison


def run_agent_turn(
    user_message: str, 
    history: list, 
    kc_model,
    progress_callback=None,
    result_callback=None
):
    """
    Run one turn of the agent.
    """
    if not is_restaurant_query(user_message):
        return run_general_chat(user_message, history)

    return run_restaurant_analysis(user_message, history, kc_model, progress_callback, result_callback)


def run_general_chat(user_message: str, history: list):
    """Handle general conversation without using tools."""
    history.append({
        "role": "user",
        "content": user_message,
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hi, I'm ready to chat."},
        {
            "role": "assistant",
            "content": (
                "Hello. I'm ReviewLens. I can chat normally or help analyze Korean restaurant reviews."
            ),
        },
    ] + history

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )

    answer = enforce_language_policy(response["message"]["content"])
    history.append({"role": "assistant", "content": answer})
    return history, answer, None


def run_restaurant_analysis(
    user_message: str, 
    history: list, 
    kc_model,
    progress_callback=None,
    result_callback=None
):
    """Handle restaurant analysis with tools."""
    history.append({
        "role": "user",
        "content": (
            f"[{LANGUAGE_POLICY} Keep Korean only for restaurant names or short quoted review text.]\n\n"
            f"{user_message}"
        ),
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": (
                "Understood. I will answer in English and keep Korean only for restaurant names or short quoted review text."
            ),
        },
    ] + history

    structured_result = None
    step = 0
    max_steps = 5

    while step < max_steps:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOLS,
        )

        if not response["message"].get("tool_calls"):
            answer = enforce_language_policy(response["message"]["content"])
            history.append({"role": "assistant", "content": answer})
            return history, answer, structured_result

        tool_call = response["message"]["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]

        result = run_tool(
            tool_name, 
            tool_args, 
            kc_model,
            progress_callback=progress_callback,
            result_callback=result_callback
        )

        # Parse and store structured result
        try:
            parsed_result = json.loads(result)
            
            # Store the result immediately for all tool types
            if isinstance(parsed_result, list) and len(parsed_result) == 2:
                structured_result = parsed_result
            elif isinstance(parsed_result, dict):
                if any(key in parsed_result for key in ["FOOD", "PRICE", "SERVICE", "AMBIENCE"]):
                    structured_result = parsed_result
                    
        except Exception as e:
            print(f"Error parsing result: {e}")

        messages.append(response["message"])
        messages.append({"role": "tool", "content": result})
        step += 1

    answer = "I analyzed the reviews for you. Check the results panel on the right."
    history.append({"role": "assistant", "content": answer})
    return history, answer, structured_result