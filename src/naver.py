from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
import requests
import re


def clean_html(text: str) -> str:
    """Remove HTML tags and decode special characters."""
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    return text.strip()

def search_restaurant_reviews(restaurant_name: str, display: int = 30) -> list:
    """Search for reviews of a specific restaurant."""
    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": f"{restaurant_name} 리뷰 후기",  # more specific
        "display": display,
        "sort": "date"
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    items = data.get("items", [])
    
    reviews = []
    for item in items:
        description = clean_html(item["description"])
        title = clean_html(item["title"])
        if len(description) > 20:  # skip very short descriptions
            reviews.append({
                "title": title,
                "description": description,
                "link": item["link"]
            })
    
    return reviews
