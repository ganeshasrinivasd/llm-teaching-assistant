"""
Query Enhancement Service

Uses LLM to enhance user queries for better retrieval:
- Expand with related terms
- Detect user intent (explain, compare, simplify)
- Infer difficulty level
"""

import json
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class EnhancedQuery(BaseModel):
    """Enhanced query with metadata."""
    original: str
    enhanced: str
    intent: str  # "explain", "compare", "summarize", "simplify", "deep_dive"
    detected_difficulty: str  # "beginner", "intermediate", "advanced"
    key_concepts: list[str]
    is_comparison: bool = False


class QueryService:
    """
    Enhance user queries for better retrieval.
    
    Features:
    - Query expansion with related terms
    - Intent detection (explain vs compare vs simplify)
    - Difficulty inference from phrasing
    - Key concept extraction
    """
    
    SYSTEM_PROMPT = """You are a query enhancement system for an academic paper search engine.

Given a user query about machine learning, AI, or computer science, analyze it and output JSON with:

{
    "enhanced": "expanded query with related technical terms for better search",
    "intent": "one of: explain, compare, summarize, simplify, deep_dive",
    "detected_difficulty": "one of: beginner, intermediate, advanced",
    "key_concepts": ["list", "of", "3-5", "key", "concepts"],
    "is_comparison": true/false
}

Intent Detection Rules:
- "ELI5", "simply", "basics", "beginner", "intro" → intent: "simplify", difficulty: "beginner"
- "Compare X vs Y", "difference between" → intent: "compare", is_comparison: true
- "Deep dive", "in-depth", "technical details" → intent: "deep_dive", difficulty: "advanced"
- "How does X work", "What is X" → intent: "explain"
- "Summarize", "overview", "brief" → intent: "summarize"

Query Enhancement Rules:
- Add related technical terms that would appear in academic papers
- Include synonyms and related concepts
- Keep the enhanced query concise (under 15 words)

Examples:
- "ELI5 attention" → enhanced: "attention mechanism transformer neural network basics introduction"
- "BERT vs GPT" → enhanced: "BERT GPT language model comparison pretraining architecture"
- "How do transformers work" → enhanced: "transformer architecture self-attention encoder decoder mechanism"
"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
    
    def enhance_query(self, query: str) -> EnhancedQuery:
        """
        Enhance a user query for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            EnhancedQuery with expanded terms and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            enhanced = EnhancedQuery(
                original=query,
                enhanced=result.get("enhanced", query),
                intent=result.get("intent", "explain"),
                detected_difficulty=result.get("detected_difficulty", "beginner"),
                key_concepts=result.get("key_concepts", []),
                is_comparison=result.get("is_comparison", False)
            )
            
            logger.info(
                f"Enhanced query: '{query[:30]}...' → intent={enhanced.intent}, "
                f"difficulty={enhanced.detected_difficulty}"
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            # Return basic enhancement on failure
            return EnhancedQuery(
                original=query,
                enhanced=query,
                intent="explain",
                detected_difficulty="beginner",
                key_concepts=query.split()[:5]
            )
    
    def quick_intent_detection(self, query: str) -> tuple[str, str]:
        """
        Fast intent detection without LLM call.
        
        Returns:
            (intent, difficulty)
        """
        query_lower = query.lower()
        
        # Simplify indicators
        if any(word in query_lower for word in ["eli5", "simple", "basics", "beginner", "intro"]):
            return "simplify", "beginner"
        
        # Comparison indicators
        if any(word in query_lower for word in [" vs ", " versus ", "compare", "difference"]):
            return "compare", "intermediate"
        
        # Deep dive indicators
        if any(word in query_lower for word in ["deep dive", "in-depth", "technical", "advanced"]):
            return "deep_dive", "advanced"
        
        # Summary indicators
        if any(word in query_lower for word in ["summarize", "overview", "brief", "tldr"]):
            return "summarize", "beginner"
        
        # Default
        return "explain", "beginner"


# Singleton instance
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get singleton Query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service
