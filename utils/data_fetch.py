from newspaper import Article
from datetime import datetime
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

class ArticleMetadata(BaseModel):
    id: str
    title: str
    description: str
    url: HttpUrl
    published_at: datetime
    source_name: str
    source_url: HttpUrl
    country: str
    language: str


def extract_full_text_with_gnews(raw: dict):
    try:
        article_url = raw.get("url")
        if not article_url:
            return None, None  

        article = Article(article_url, language="en")
        article.download()
        article.parse()
        return article.text, ArticleMetadata(
                id=raw["id"],
                title=article.title,
                description=raw.get("description", ""),
                url=raw["url"],
                published_at=datetime.fromisoformat(raw["publishedAt"].replace("Z", "+00:00")),
                source_name=raw["source"]["name"],
                source_url=raw["source"]["url"],
                country=raw["source"]["country"],
                language=raw["lang"]
            )
    except Exception as e:
        print(f"[WARN] Skipping article due to error: {e}")
        return None, None   #