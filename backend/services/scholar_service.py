"""
Semantic Scholar API Service

Fetches academic papers dynamically when:
- User query doesn't match existing papers well
- Need to expand knowledge base

API Docs: https://api.semanticscholar.org/
Rate Limit: 100 requests per 5 minutes (free tier)
"""

import httpx
from typing import Optional
from datetime import date

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class ScholarService:
    """
    Fetch academic papers from Semantic Scholar API.
    
    Features:
    - Free API (no scraping needed)
    - Structured data with abstracts
    - ArXiv integration
    - Rate limiting built-in
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.Client(timeout=30.0)
        self._daily_fetch_count = 0
        self._last_reset_date = date.today()
    
    def _check_daily_limit(self) -> bool:
        """Check and reset daily limit if needed."""
        today = date.today()
        if today > self._last_reset_date:
            self._daily_fetch_count = 0
            self._last_reset_date = today
        
        return self._daily_fetch_count < self.settings.max_daily_fetches
    
    def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_min: Optional[int] = None
    ) -> list[dict]:
        """
        Search for papers matching a query.
        
        Args:
            query: Search query
            limit: Max results (max 100)
            year_min: Minimum publication year
        
        Returns:
            List of paper objects with abstracts
        """
        if not self._check_daily_limit():
            logger.warning(f"Daily fetch limit reached ({self.settings.max_daily_fetches})")
            return []
        
        fields = [
            "paperId",
            "title",
            "abstract",
            "url",
            "year",
            "authors",
            "citationCount",
            "externalIds",
            "openAccessPdf"
        ]
        
        try:
            params = {
                "query": query,
                "limit": min(limit, 100),
                "fields": ",".join(fields)
            }
            
            if year_min:
                params["year"] = f"{year_min}-"
            
            # Add API key if available (higher rate limits)
            headers = {}
            if self.settings.semantic_scholar_api_key:
                headers["x-api-key"] = self.settings.semantic_scholar_api_key
            
            response = self.client.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            papers = data.get("data", [])
            
            # Filter: must have abstract for embedding
            papers_with_abstracts = [
                p for p in papers
                if p.get("abstract") and len(p.get("abstract", "")) > 100
            ]
            
            self._daily_fetch_count += 1
            logger.info(
                f"Fetched {len(papers_with_abstracts)}/{len(papers)} papers "
                f"for query: '{query[:50]}...' (daily: {self._daily_fetch_count}/{self.settings.max_daily_fetches})"
            )
            
            return papers_with_abstracts
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit, backing off")
            else:
                logger.error(f"Semantic Scholar API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Semantic Scholar error: {e}")
            return []
    
    def get_paper_by_id(self, paper_id: str) -> Optional[dict]:
        """Get a specific paper by Semantic Scholar ID."""
        try:
            response = self.client.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={
                    "fields": "paperId,title,abstract,url,year,authors,externalIds,openAccessPdf"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return None
    
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """Get a paper by ArXiv ID."""
        try:
            response = self.client.get(
                f"{self.BASE_URL}/paper/arXiv:{arxiv_id}",
                params={
                    "fields": "paperId,title,abstract,url,year,authors,externalIds"
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return None
    
    @staticmethod
    def extract_arxiv_id(paper: dict) -> Optional[str]:
        """Extract ArXiv ID from paper data."""
        external_ids = paper.get("externalIds", {})
        return external_ids.get("ArXiv")
    
    @staticmethod
    def get_arxiv_url(paper: dict) -> Optional[str]:
        """Get ArXiv URL if available."""
        arxiv_id = ScholarService.extract_arxiv_id(paper)
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
        return paper.get("url")
    
    @staticmethod
    def get_pdf_url(paper: dict) -> Optional[str]:
        """Get PDF URL if available."""
        # Try open access PDF first
        open_access = paper.get("openAccessPdf", {})
        if open_access and open_access.get("url"):
            return open_access["url"]
        
        # Try ArXiv PDF
        arxiv_id = ScholarService.extract_arxiv_id(paper)
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return None
    
    def get_daily_stats(self) -> dict:
        """Get daily usage statistics."""
        return {
            "fetches_today": self._daily_fetch_count,
            "max_daily": self.settings.max_daily_fetches,
            "remaining": self.settings.max_daily_fetches - self._daily_fetch_count
        }


# Singleton instance
_scholar_service: Optional[ScholarService] = None


def get_scholar_service() -> ScholarService:
    """Get singleton Scholar service instance."""
    global _scholar_service
    if _scholar_service is None:
        _scholar_service = ScholarService()
    return _scholar_service
