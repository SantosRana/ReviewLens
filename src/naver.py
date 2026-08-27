# src/naver.py
import os
import html
import re
import logging
import requests
from typing import List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class NaverBlogSearch:
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.api_url = "https://openapi.naver.com/v1/search/blog.json"

    def _clean_html(self, text: str) -> str:
        """Decodes HTML entities and strips HTML markup tags."""
        if not text:
            return ""
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    def fetch_reviews(self, restaurant_name: str, display_count: int = 20) -> List[str]:
        """
        Queries Naver Blog Search API for restaurant reviews and returns 
        a cleaned list of string review descriptions for ABSA processing.
        """
        if not self.client_id or not self.client_secret:
            logger.error("Naver API credentials (NAVER_CLIENT_ID / NAVER_CLIENT_SECRET) missing.")
            return []

        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {
            "query": f"{restaurant_name} 맛집",
            "display": min(display_count, 100),  # Naver API permits max 100 items per request
            "sort": "sim"  # Sort by similarity/relevance
        }

        try:
            response = requests.get(self.api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Naver API request failed: {e}")
            return []
        except ValueError:
            logger.error("Naver API returned invalid JSON.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Naver fetch: {e}")
            return []

        if "errorMessage" in data:
            logger.error(f"Naver API error response: {data.get('errorMessage')}")
            return []

        raw_items = data.get("items", [])
        reviews: List[str] = []

        for item in raw_items:
            title = self._clean_html(item.get("title", ""))
            description = self._clean_html(item.get("description", ""))
            
            # Combine title + snippet for maximum context window exposure
            combined_text = f"{title}. {description}".strip()

            # Filter out ultra-short or empty snippet noise
            if len(description) > 20:
                reviews.append(combined_text)

        return reviews