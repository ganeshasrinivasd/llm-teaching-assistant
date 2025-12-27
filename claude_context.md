# Project Context for Claude

Generated: Fri Dec 26 20:02:00 PST 2025

## 1. Project Structure
```
./frontend/tsconfig.node.json
./frontend/node_modules/queue-microtask/package.json
./frontend/node_modules/queue-microtask/index.d.ts
./frontend/node_modules/is-plain-obj/package.json
./frontend/node_modules/is-plain-obj/index.d.ts
./frontend/node_modules/tinyglobby/node_modules/picomatch/package.json
./frontend/node_modules/tinyglobby/node_modules/fdir/package.json
./frontend/node_modules/tinyglobby/package.json
./frontend/node_modules/@alloc/quick-lru/package.json
./frontend/node_modules/@alloc/quick-lru/index.d.ts
./frontend/node_modules/reusify/package.json
./frontend/node_modules/reusify/reusify.d.ts
./frontend/node_modules/reusify/tsconfig.json
./frontend/node_modules/zwitch/package.json
./frontend/node_modules/zwitch/index.d.ts
./frontend/node_modules/jsesc/package.json
./frontend/node_modules/pirates/package.json
./frontend/node_modules/pirates/index.d.ts
./frontend/node_modules/@types/ms/package.json
./frontend/node_modules/@types/ms/index.d.ts
./frontend/node_modules/@types/hast/package.json
./frontend/node_modules/@types/hast/index.d.ts
./frontend/node_modules/@types/babel__template/package.json
./frontend/node_modules/@types/babel__template/index.d.ts
./frontend/node_modules/@types/react-dom/test-utils/index.d.ts
./frontend/node_modules/@types/react-dom/server.d.ts
./frontend/node_modules/@types/react-dom/canary.d.ts
./frontend/node_modules/@types/react-dom/experimental.d.ts
./frontend/node_modules/@types/react-dom/package.json
./frontend/node_modules/@types/react-dom/index.d.ts
./frontend/node_modules/@types/react-dom/client.d.ts
./frontend/node_modules/@types/babel__generator/package.json
./frontend/node_modules/@types/babel__generator/index.d.ts
./frontend/node_modules/@types/babel__traverse/package.json
./frontend/node_modules/@types/babel__traverse/index.d.ts
./frontend/node_modules/@types/prop-types/package.json
./frontend/node_modules/@types/prop-types/index.d.ts
./frontend/node_modules/@types/mdast/package.json
./frontend/node_modules/@types/mdast/index.d.ts
./frontend/node_modules/@types/estree/flow.d.ts
./frontend/node_modules/@types/estree/package.json
./frontend/node_modules/@types/estree/index.d.ts
./frontend/node_modules/@types/unist/package.json
./frontend/node_modules/@types/unist/index.d.ts
./frontend/node_modules/@types/babel__core/package.json
./frontend/node_modules/@types/babel__core/index.d.ts
./frontend/node_modules/@types/react/jsx-dev-runtime.d.ts
./frontend/node_modules/@types/react/jsx-runtime.d.ts
./frontend/node_modules/@types/react/canary.d.ts
./frontend/node_modules/@types/react/experimental.d.ts
./frontend/node_modules/@types/react/package.json
./frontend/node_modules/@types/react/global.d.ts
./frontend/node_modules/@types/react/ts5.0/jsx-dev-runtime.d.ts
./frontend/node_modules/@types/react/ts5.0/jsx-runtime.d.ts
./frontend/node_modules/@types/react/ts5.0/canary.d.ts
./frontend/node_modules/@types/react/ts5.0/experimental.d.ts
./frontend/node_modules/@types/react/ts5.0/global.d.ts
./frontend/node_modules/@types/react/ts5.0/index.d.ts
./frontend/node_modules/@types/react/index.d.ts
./frontend/node_modules/@types/debug/package.json
./frontend/node_modules/@types/debug/index.d.ts
./frontend/node_modules/@types/estree-jsx/package.json
./frontend/node_modules/@types/estree-jsx/index.d.ts
./frontend/node_modules/micromark-core-commonmark/package.json
./frontend/node_modules/micromark-core-commonmark/lib/content.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/html-flow.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/character-reference.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/heading-atx.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/label-start-link.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/setext-underline.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/label-end.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/autolink.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/code-fenced.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/block-quote.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/line-ending.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/character-escape.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/html-text.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/label-start-image.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/code-text.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/attention.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/list.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/hard-break-escape.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/code-indented.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/thematic-break.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/definition.d.ts
./frontend/node_modules/micromark-core-commonmark/lib/blank-line.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/content.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/html-flow.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/character-reference.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/heading-atx.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/label-start-link.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/setext-underline.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/label-end.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/autolink.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/code-fenced.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/block-quote.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/line-ending.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/character-escape.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/html-text.d.ts
./frontend/node_modules/micromark-core-commonmark/dev/lib/label-start-image.d.ts
```

## 2. Backend Code

### backend/core/config.py
```python
"""
Configuration Management

Centralized configuration with environment variable support,
validation, and sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


# Get the project root directory (where this config.py is located, go up one level)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App Info
    app_name: str = "LLM Teaching Assistant"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # OpenAI Settings
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    lesson_model: str = "gpt-4o-mini"  # Cheaper model for lesson generation
    
    # GROBID Settings
    grobid_url: str = "https://kermitt2-grobid.hf.space"
    grobid_timeout: int = 120  # seconds
    use_grobid: bool = True  # Can disable for abstract-only mode
    
    # File Paths - Now absolute paths based on PROJECT_ROOT
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    faiss_index_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss" / "papers.index")
    urls_json_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss" / "urls.json")
    cache_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    
    # Cache Settings
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours in seconds
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30  # requests per minute
    rate_limit_window: int = 60  # seconds
    
    # LeetCode Settings
    leetcode_difficulties: list[str] = ["Medium", "Hard"]
    leetcode_allow_premium: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your-api-key-here":
            raise ValueError("Valid OpenAI API key is required")
        return v
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience function
def get_config() -> Settings:
    """Alias for get_settings."""
    return get_settings()
```

### backend/services/__init__.py
```python
"""
Services Package

Business logic layer for the application.
"""

from .cache_service import CacheService, get_cache_service
from .embedding_service import EmbeddingService, get_embedding_service
from .paper_service import PaperService, get_paper_service
from .lesson_service import LessonService, get_lesson_service
from .leetcode_service import LeetCodeService, get_leetcode_service
from .teaching_service import TeachingService, get_teaching_service

__all__ = [
    # Cache
    "CacheService",
    "get_cache_service",
    # Embedding
    "EmbeddingService",
    "get_embedding_service",
    # Paper
    "PaperService",
    "get_paper_service",
    # Lesson
    "LessonService",
    "get_lesson_service",
    # LeetCode
    "LeetCodeService",
    "get_leetcode_service",
    # Teaching
    "TeachingService",
    "get_teaching_service",
]
```

### backend/services/cache_service.py
```python
"""
Cache Service

File-based and in-memory caching for improved performance.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic
from functools import lru_cache
from pydantic import BaseModel

from core.config import get_settings
from core.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class CacheEntry(BaseModel, Generic[T]):
    """A cached entry with metadata."""
    
    key: str
    value: Any
    created_at: float
    ttl: int
    
    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl)


class CacheService:
    """
    Hybrid cache service with in-memory and file-based storage.
    
    - Hot data stays in memory (LRU cache)
    - Cold data persists to disk
    - Automatic TTL expiration
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 86400):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.default_ttl = default_ttl
        self.enabled = settings.cache_enabled
        
        # In-memory cache
        self._memory_cache: dict[str, CacheEntry] = {}
        self._max_memory_items = 100
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache service initialized: dir={self.cache_dir}, enabled={self.enabled}")
    
    def _make_key(self, namespace: str, identifier: str) -> str:
        """Create a cache key from namespace and identifier."""
        raw = f"{namespace}:{identifier}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            namespace: Cache namespace (e.g., "lessons", "papers")
            identifier: Unique identifier within namespace
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._make_key(namespace, identifier)
        
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                logger.debug(f"Cache hit (memory): {namespace}:{identifier[:20]}")
                return entry.value
            else:
                del self._memory_cache[key]
        
        # Check file cache
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    entry = CacheEntry(**data)
                    
                    if not entry.is_expired:
                        # Promote to memory cache
                        self._memory_cache[key] = entry
                        self._evict_if_needed()
                        logger.debug(f"Cache hit (file): {namespace}:{identifier[:20]}")
                        return entry.value
                    else:
                        # Remove expired file
                        file_path.unlink()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        logger.debug(f"Cache miss: {namespace}:{identifier[:20]}")
        return None
    
    def set(
        self,
        namespace: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            namespace: Cache namespace
            identifier: Unique identifier
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (default: 24 hours)
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        key = self._make_key(namespace, identifier)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
        
        # Save to memory
        self._memory_cache[key] = entry
        self._evict_if_needed()
        
        # Save to file
        try:
            file_path = self._get_file_path(key)
            with open(file_path, "w") as f:
                json.dump(entry.model_dump(), f)
            logger.debug(f"Cached: {namespace}:{identifier[:20]}")
            return True
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def delete(self, namespace: str, identifier: str) -> bool:
        """Delete a cached value."""
        key = self._make_key(namespace, identifier)
        
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Remove from file
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        
        return False
    
    def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            namespace: If provided, only clear this namespace
            
        Returns:
            Number of entries cleared
        """
        count = 0
        
        if namespace is None:
            # Clear all
            count = len(self._memory_cache)
            self._memory_cache.clear()
            
            for file_path in self.cache_dir.glob("*.json"):
                file_path.unlink()
                count += 1
        
        logger.info(f"Cache cleared: {count} entries")
        return count
    
    def _evict_if_needed(self):
        """Evict oldest entries if memory cache is too large."""
        while len(self._memory_cache) > self._max_memory_items:
            # Remove oldest entry
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].created_at
            )
            del self._memory_cache[oldest_key]
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        file_count = len(list(self.cache_dir.glob("*.json")))
        
        return {
            "enabled": self.enabled,
            "memory_entries": len(self._memory_cache),
            "file_entries": file_count,
            "cache_dir": str(self.cache_dir),
            "default_ttl": self.default_ttl
        }


# Singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
```

### backend/services/embedding_service.py
```python
"""
Embedding Service

Vector embedding generation and FAISS index management.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import faiss
from openai import OpenAI

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import EmbeddingError, IndexNotFoundError

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings and managing FAISS index.
    
    Features:
    - Batch embedding generation
    - FAISS index creation and search
    - URL mapping management
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model
        
        self._index: Optional[faiss.Index] = None
        self._urls: Optional[list[str]] = None
        
        logger.info(f"Embedding service initialized: model={self.model}")
    
    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index, loading if necessary."""
        if self._index is None:
            self.load_index()
        return self._index
    
    @property
    def urls(self) -> list[str]:
        """Get the URL list, loading if necessary."""
        if self._urls is None:
            self.load_urls()
        return self._urls
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = np.array(response.data[0].embedding, dtype="float32")
            return embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise EmbeddingError(f"Failed to create embedding: {e}")
    
    def create_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise EmbeddingError(f"Failed to create batch embeddings: {e}")
        
        return np.array(all_embeddings, dtype="float32")
    
    def build_index(self, embeddings: np.ndarray, urls: list[str]) -> None:
        """
        Build and save FAISS index.
        
        Args:
            embeddings: Array of embeddings
            urls: Corresponding URLs
        """
        if len(embeddings) != len(urls):
            raise ValueError("Embeddings and URLs must have same length")
        
        # Create FAISS index (L2 distance)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save index
        self.settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.settings.faiss_index_path))
        
        # Save URLs
        with open(self.settings.urls_json_path, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, indent=2)
        
        self._index = index
        self._urls = urls
        
        logger.info(f"Built FAISS index with {len(urls)} entries")
    
    def load_index(self) -> None:
        """Load FAISS index from disk."""
        index_path = self.settings.faiss_index_path
        
        if not index_path.exists():
            raise IndexNotFoundError(str(index_path))
        
        self._index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")
    
    def load_urls(self) -> None:
        """Load URLs from disk."""
        urls_path = self.settings.urls_json_path
        
        if not urls_path.exists():
            raise IndexNotFoundError(str(urls_path))
        
        with open(urls_path, "r", encoding="utf-8") as f:
            self._urls = json.load(f)
        
        logger.info(f"Loaded {len(self._urls)} URLs")
    
    def search(self, query: str, k: int = 1) -> list[tuple[int, float, str]]:
        """
        Search for similar papers.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (index, distance, url) tuples
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            url = self.urls[idx] if idx < len(self.urls) else ""
            results.append((idx, dist, url))
        
        logger.debug(f"Search for '{query[:30]}...' returned {len(results)} results")
        return results
    
    def search_by_embedding(self, embedding: np.ndarray, k: int = 1) -> list[tuple[int, float, str]]:
        """
        Search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding
            k: Number of results
            
        Returns:
            List of (index, distance, url) tuples
        """
        embedding = embedding.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            url = self.urls[idx] if idx < len(self.urls) else ""
            results.append((idx, dist, url))
        
        return results
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "model": self.model,
            "index_loaded": self._index is not None,
            "index_size": self._index.ntotal if self._index else 0,
            "urls_loaded": self._urls is not None,
            "urls_count": len(self._urls) if self._urls else 0,
            "index_path": str(self.settings.faiss_index_path),
        }


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
```

### backend/services/leetcode_service.py
```python
"""
LeetCode Service

Fetches coding problems from LeetCode for interview practice.
"""

import random
import requests
import bs4
from typing import Optional

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LeetCodeError
from models.problem import (
    LeetCodeProblem,
    ProblemDifficulty,
    ProblemRequest,
    ProblemCatalogEntry,
)
from services.cache_service import get_cache_service

logger = get_logger(__name__)


# LeetCode API constants
LEETCODE_CATALOG_URL = "https://leetcode.com/api/problems/algorithms/"
LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
LEETCODE_GRAPHQL_QUERY = """
query questionData($titleSlug: String!) {
    question(titleSlug: $titleSlug) {
        content
        hints
        topicTags { name }
    }
}
"""


class LeetCodeService:
    """
    Service for fetching LeetCode problems.
    
    Features:
    - Catalog fetching with caching
    - Random problem selection by difficulty
    - Problem statement retrieval
    - Topic filtering
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache = get_cache_service()
        self._catalog: Optional[list[ProblemCatalogEntry]] = None
        
        logger.info("LeetCode service initialized")
    
    def get_random_problem(self, request: Optional[ProblemRequest] = None) -> LeetCodeProblem:
        """
        Get a random LeetCode problem.
        
        Args:
            request: Problem selection criteria
            
        Returns:
            Random problem matching criteria
        """
        request = request or ProblemRequest()
        
        # Get catalog
        catalog = self._get_catalog()
        
        # Filter problems
        filtered = self._filter_problems(catalog, request)
        
        if not filtered:
            raise LeetCodeError("No problems match the specified criteria")
        
        # Select random problem
        selected = random.choice(filtered)
        
        # Fetch full problem
        return self._fetch_problem(selected)
    
    def get_problem_by_slug(self, slug: str) -> LeetCodeProblem:
        """
        Get a specific problem by slug.
        
        Args:
            slug: Problem URL slug (e.g., "two-sum")
            
        Returns:
            The problem
        """
        # Check cache
        cached = self.cache.get("leetcode_problems", slug)
        if cached:
            logger.debug(f"LeetCode cache hit: {slug}")
            return LeetCodeProblem(**cached)
        
        # Find in catalog
        catalog = self._get_catalog()
        entry = next((p for p in catalog if p.slug == slug), None)
        
        if not entry:
            raise LeetCodeError(f"Problem not found: {slug}")
        
        return self._fetch_problem(entry)
    
    def _get_catalog(self) -> list[ProblemCatalogEntry]:
        """Get the problem catalog, with caching."""
        if self._catalog is not None:
            return self._catalog
        
        # Check cache
        cached = self.cache.get("leetcode", "catalog")
        if cached:
            self._catalog = [ProblemCatalogEntry(**p) for p in cached]
            logger.debug(f"Loaded catalog from cache: {len(self._catalog)} problems")
            return self._catalog
        
        # Fetch from API
        self._catalog = self._fetch_catalog()
        
        # Cache for 24 hours
        self.cache.set(
            "leetcode",
            "catalog",
            [p.model_dump() for p in self._catalog],
            ttl=86400
        )
        
        return self._catalog
    
    def _fetch_catalog(self) -> list[ProblemCatalogEntry]:
        """Fetch the problem catalog from LeetCode API."""
        logger.info("Fetching LeetCode catalog...")
        
        try:
            response = requests.get(LEETCODE_CATALOG_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise LeetCodeError(f"Failed to fetch catalog: {e}")
        
        difficulty_map = {1: ProblemDifficulty.EASY, 2: ProblemDifficulty.MEDIUM, 3: ProblemDifficulty.HARD}
        
        catalog = []
        for problem in data.get("stat_status_pairs", []):
            stat = problem.get("stat", {})
            diff = problem.get("difficulty", {})
            
            catalog.append(ProblemCatalogEntry(
                slug=stat.get("question__title_slug", ""),
                title=stat.get("question__title", ""),
                difficulty=difficulty_map.get(diff.get("level", 2), ProblemDifficulty.MEDIUM),
                paid_only=problem.get("paid_only", False),
                acceptance_rate=stat.get("total_acs", 0) / max(stat.get("total_submitted", 1), 1) * 100
            ))
        
        logger.info(f"Fetched {len(catalog)} problems")
        return catalog
    
    def _filter_problems(
        self,
        catalog: list[ProblemCatalogEntry],
        request: ProblemRequest
    ) -> list[ProblemCatalogEntry]:
        """Filter problems based on request criteria."""
        filtered = []
        
        for problem in catalog:
            # Filter by premium
            if request.exclude_premium and problem.paid_only:
                continue
            
            # Filter by difficulty
            if problem.difficulty not in request.difficulties:
                continue
            
            filtered.append(problem)
        
        return filtered
    
    def _fetch_problem(self, entry: ProblemCatalogEntry) -> LeetCodeProblem:
        """Fetch full problem details."""
        # Check cache
        cached = self.cache.get("leetcode_problems", entry.slug)
        if cached:
            return LeetCodeProblem(**cached)
        
        # Fetch via GraphQL
        try:
            response = requests.post(
                LEETCODE_GRAPHQL_URL,
                json={
                    "query": LEETCODE_GRAPHQL_QUERY,
                    "variables": {"titleSlug": entry.slug}
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise LeetCodeError(f"Failed to fetch problem {entry.slug}: {e}")
        
        question = data.get("data", {}).get("question", {})
        
        if not question:
            raise LeetCodeError(f"Problem not found: {entry.slug}")
        
        # Parse HTML content
        html_content = question.get("content", "")
        statement = self._parse_html(html_content)
        
        # Extract topics
        topics = [tag.get("name", "") for tag in question.get("topicTags", [])]
        
        # Extract hints
        hints = question.get("hints", [])
        
        problem = LeetCodeProblem(
            title=entry.title,
            slug=entry.slug,
            difficulty=entry.difficulty,
            statement=statement,
            topics=topics,
            hints=hints,
            acceptance_rate=entry.acceptance_rate
        )
        
        # Cache problem
        self.cache.set("leetcode_problems", entry.slug, problem.model_dump())
        
        return problem
    
    def _parse_html(self, html: str) -> str:
        """Parse HTML content to clean text."""
        if not html:
            return ""
        
        soup = bs4.BeautifulSoup(html, "html.parser")
        return soup.get_text("\n").strip()
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        catalog_size = len(self._catalog) if self._catalog else 0
        
        return {
            "catalog_loaded": self._catalog is not None,
            "catalog_size": catalog_size,
            "cache_stats": self.cache.get_stats()
        }


# Singleton instance
_leetcode_service: Optional[LeetCodeService] = None


def get_leetcode_service() -> LeetCodeService:
    """Get the global LeetCode service instance."""
    global _leetcode_service
    if _leetcode_service is None:
        _leetcode_service = LeetCodeService()
    return _leetcode_service
```

### backend/services/lesson_service.py
```python
"""
Lesson Generation Service

Converts research paper sections into beginner-friendly lessons.
"""

import time
import asyncio
from typing import Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import LessonGenerationError
from models.paper import ParsedPaper, PaperSection
from models.lesson import (
    LessonFragment,
    FullLesson,
    LessonRequest,
    LessonDifficulty,
    StreamingLessonChunk,
)
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class LessonService:
    """
    Service for generating lessons from paper sections.
    
    Features:
    - Beginner-friendly explanations
    - Step-by-step math breakdowns
    - Smooth section transitions
    - Streaming support
    - Caching
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.cache = get_cache_service()
        
        logger.info(f"Lesson service initialized: model={self.settings.lesson_model}")
    
    def generate_lesson(
        self,
        paper: ParsedPaper,
        request: LessonRequest
    ) -> FullLesson:
        """
        Generate a full lesson from a parsed paper.
        
        Args:
            paper: Parsed paper with sections
            request: Lesson generation request
            
        Returns:
            Complete lesson with all fragments
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{paper.metadata.arxiv_id}:{request.difficulty}"
        cached = self.cache.get("lessons", cache_key)
        if cached:
            logger.info(f"Lesson cache hit: {paper.metadata.arxiv_id}")
            return FullLesson(**cached)
        
        # Generate fragments
        fragments = []
        sections = paper.sections
        
        if request.max_sections:
            sections = sections[:request.max_sections]
        
        for i, section in enumerate(sections):
            next_section = sections[i + 1] if i + 1 < len(sections) else None
            
            fragment = self._generate_fragment(
                section=section,
                next_section_name=next_section.name if next_section else None,
                request=request,
                order=i
            )
            fragments.append(fragment)
            logger.debug(f"Generated fragment {i + 1}/{len(sections)}: {section.name}")
        
        # Create full lesson
        lesson = FullLesson(
            paper_id=paper.metadata.arxiv_id,
            paper_title=paper.metadata.title,
            paper_url=str(paper.metadata.url),
            query=request.query,
            fragments=fragments,
            difficulty=request.difficulty,
            generation_time_seconds=time.time() - start_time
        )
        
        # Cache result
        self.cache.set("lessons", cache_key, lesson.model_dump(mode='json'))
        
        logger.info(
            f"Generated lesson for {paper.metadata.arxiv_id}: "
            f"{len(fragments)} sections, {lesson.total_read_time} min read time"
        )
        
        return lesson
    
    async def generate_lesson_streaming(
        self,
        paper: ParsedPaper,
        request: LessonRequest
    ) -> AsyncGenerator[StreamingLessonChunk, None]:
        """
        Generate lesson with streaming responses.
        
        Yields chunks as sections are generated.
        """
        start_time = time.time()
        
        # Send metadata first
        yield StreamingLessonChunk(
            type="metadata",
            data={
                "paper_id": paper.metadata.arxiv_id,
                "paper_title": paper.metadata.title,
                "paper_url": str(paper.metadata.url),
                "total_sections": len(paper.sections),
            }
        )
        
        sections = paper.sections
        if request.max_sections:
            sections = sections[:request.max_sections]
        
        for i, section in enumerate(sections):
            try:
                next_section = sections[i + 1] if i + 1 < len(sections) else None
                
                content = await self._generate_fragment_async(
                    section=section,
                    next_section_name=next_section.name if next_section else None,
                    request=request
                )
                
                yield StreamingLessonChunk(
                    type="section",
                    data={
                        "name": section.name,
                        "content": content,
                        "order": i,
                        "progress": (i + 1) / len(sections)
                    }
                )
            except Exception as e:
                logger.error(f"Error generating section {section.name}: {e}")
                yield StreamingLessonChunk(
                    type="error",
                    data={"message": f"Failed to generate section: {section.name}"}
                )
        
        # Send completion
        yield StreamingLessonChunk(
            type="done",
            data={
                "total_time_seconds": time.time() - start_time,
                "sections_generated": len(sections)
            }
        )
    
    def _generate_fragment(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest,
        order: int
    ) -> LessonFragment:
        """Generate a single lesson fragment."""
        prompt = self._build_prompt(section, next_section_name, request)
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.lesson_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            raise LessonGenerationError(f"Failed to generate lesson: {e}", section.name)
        
        return LessonFragment(
            section_name=section.name,
            content=content,
            order=order,
            has_math=self._contains_math(content),
            has_code=self._contains_code(content)
        )
    
    async def _generate_fragment_async(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest
    ) -> str:
        """Generate a single lesson fragment asynchronously."""
        prompt = self._build_prompt(section, next_section_name, request)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.settings.lesson_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LessonGenerationError(f"Failed to generate lesson: {e}", section.name)
    
    def _build_prompt(
        self,
        section: PaperSection,
        next_section_name: Optional[str],
        request: LessonRequest
    ) -> str:
        """Build the prompt for lesson generation."""
        difficulty_instructions = {
            LessonDifficulty.BEGINNER: "Use simple language, avoid jargon, and explain everything from first principles.",
            LessonDifficulty.INTERMEDIATE: "Assume basic ML/CS knowledge but explain advanced concepts clearly.",
            LessonDifficulty.ADVANCED: "Be concise and technical, focusing on nuances and advanced insights."
        }
        
        prompt = f"""You are an expert teacher converting a research paper section into a {request.difficulty.value}-friendly lesson.

Section: "{section.name}"

Content:
{section.content}

Instructions:
- {difficulty_instructions[request.difficulty]}
"""
        
        if request.include_math:
            prompt += "- Break down any mathematical concepts step by step.\n"
        else:
            prompt += "- Minimize mathematical notation, focus on intuition.\n"
        
        if request.include_examples:
            prompt += "- Include concrete examples and analogies to illustrate concepts.\n"
        
        if next_section_name:
            prompt += f'\n- End with a smooth transition to the next section: "{next_section_name}".\n'
        
        prompt += "\nGenerate the lesson fragment now:"
        
        return prompt
    
    def _contains_math(self, content: str) -> bool:
        """Check if content contains mathematical notation."""
        math_indicators = ['$', '\\frac', '\\sum', '\\int', '∑', '∫', '√', 'equation']
        return any(ind in content.lower() for ind in math_indicators)
    
    def _contains_code(self, content: str) -> bool:
        """Check if content contains code."""
        code_indicators = ['```', 'def ', 'import ', 'class ', 'function']
        return any(ind in content for ind in code_indicators)
    
    def generate_single_section_lesson(
        self,
        section_name: str,
        section_text: str,
        next_section_name: Optional[str] = None,
        difficulty: LessonDifficulty = LessonDifficulty.BEGINNER
    ) -> str:
        """
        Generate a lesson for a single section (backwards compatible).
        
        Args:
            section_name: Name of the section
            section_text: Section content
            next_section_name: Next section for transition
            difficulty: Lesson difficulty
            
        Returns:
            Generated lesson text
        """
        section = PaperSection(name=section_name, content=section_text, order=0)
        request = LessonRequest(
            query="",
            difficulty=difficulty,
            include_examples=True,
            include_math=True
        )
        
        fragment = self._generate_fragment(section, next_section_name, request, 0)
        return fragment.content


# Singleton instance
_lesson_service: Optional[LessonService] = None


def get_lesson_service() -> LessonService:
    """Get the global lesson service instance."""
    global _lesson_service
    if _lesson_service is None:
        _lesson_service = LessonService()
    return _lesson_service
```

### backend/services/paper_service.py
```python
"""
Paper Service

Paper retrieval, PDF processing, and section extraction.
"""

import re
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import (
    PaperNotFoundError,
    GROBIDError,
    ArxivError,
    PDFProcessingError,
)
from models.paper import PaperMetadata, PaperSection, ParsedPaper, PaperSearchResult
from services.embedding_service import get_embedding_service
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class PaperService:
    """
    Service for paper retrieval and processing.
    
    Features:
    - Semantic search via FAISS
    - PDF parsing via GROBID
    - Fallback to abstract-only mode
    - Caching for performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.cache = get_cache_service()
        
        logger.info(f"Paper service initialized: grobid={self.settings.grobid_url}")
    
    def search(self, query: str, top_k: int = 1) -> list[PaperSearchResult]:
        """
        Search for papers matching a query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        results = self.embedding_service.search(query, k=top_k)
        
        search_results = []
        for idx, distance, url in results:
            # Convert L2 distance to similarity score (0-1)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            
            arxiv_id = self._extract_arxiv_id(url)
            
            paper = PaperMetadata(
                arxiv_id=arxiv_id,
                title=f"Paper {arxiv_id}",  # Will be updated when fetched
                url=url
            )
            
            search_results.append(PaperSearchResult(
                paper=paper,
                similarity_score=similarity,
                index_position=idx
            ))
        
        return search_results
    
    def get_paper(self, url: str, use_grobid: bool = True) -> ParsedPaper:
        """
        Get a parsed paper from URL.
        
        Args:
            url: arXiv URL
            use_grobid: Whether to use GROBID for full parsing
            
        Returns:
            Parsed paper with sections
        """
        arxiv_id = self._extract_arxiv_id(url)
        
        # Check cache
        cache_key = f"{arxiv_id}:{'grobid' if use_grobid else 'abstract'}"
        cached = self.cache.get("papers", cache_key)
        if cached:
            logger.info(f"Paper cache hit: {arxiv_id}")
            return ParsedPaper(**cached)
        
        # Fetch metadata
        metadata = self._fetch_arxiv_metadata(arxiv_id)
        
        # Try GROBID if enabled
        sections = []
        parsing_method = "abstract"
        
        if use_grobid and self.settings.use_grobid:
            try:
                sections = self._parse_with_grobid(url)
                parsing_method = "grobid"
                logger.info(f"GROBID parsed {len(sections)} sections from {arxiv_id}")
            except Exception as e:
                logger.warning(f"GROBID failed, falling back to abstract: {e}")
        
        # Fallback: use abstract as single section
        if not sections and metadata.abstract:
            sections = [PaperSection(
                name="abstract",
                content=metadata.abstract,
                order=0
            )]
            parsing_method = "abstract"
        
        paper = ParsedPaper(
            metadata=metadata,
            sections=sections,
            parsing_method=parsing_method
        )
        
        # Cache result
        self.cache.set("papers", cache_key, paper.model_dump(mode='json'))
        
        return paper
    
    def _extract_arxiv_id(self, url: str) -> str:
        """Extract arXiv ID from URL."""
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url)
        if match:
            return match.group(1)
        
        # Try to extract from path
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if path_parts:
            return path_parts[-1].replace(".pdf", "")
        
        return url
    
    def _fetch_arxiv_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Fetch paper metadata from arXiv API."""
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ArxivError(f"Failed to fetch metadata: {e}")
        
        # Parse XML
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(response.content)
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            raise PaperNotFoundError(arxiv_id)
        
        title = entry.find('atom:title', ns)
        summary = entry.find('atom:summary', ns)
        authors = entry.findall('atom:author/atom:name', ns)
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title.text.strip() if title is not None else f"Paper {arxiv_id}",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            abstract=summary.text.strip() if summary is not None else None,
            authors=[a.text for a in authors if a.text]
        )
    
    def _parse_with_grobid(self, url: str) -> list[PaperSection]:
        """Parse paper using GROBID service."""
        # Convert to PDF URL
        if '/abs/' in url:
            arxiv_id = self._extract_arxiv_id(url)
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        else:
            pdf_url = url
        
        # Download PDF
        logger.debug(f"Downloading PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            pdf_bytes = response.content
        except requests.RequestException as e:
            raise PDFProcessingError(f"Failed to download PDF: {e}", url)
        
        # Send to GROBID
        logger.debug(f"Sending to GROBID: {self.settings.grobid_url}")
        try:
            files = {'input': ('paper.pdf', BytesIO(pdf_bytes), 'application/pdf')}
            response = requests.post(
                f'{self.settings.grobid_url}/api/processFulltextDocument',
                files=files,
                timeout=self.settings.grobid_timeout
            )
            response.raise_for_status()
            tei_xml = response.text
        except requests.RequestException as e:
            raise GROBIDError(f"GROBID processing failed: {e}")
        
        # Parse TEI XML
        return self._parse_tei_xml(tei_xml)
    
    def _parse_tei_xml(self, tei_xml: str) -> list[PaperSection]:
        """Parse TEI XML to extract sections."""
        TEI_NS = 'http://www.tei-c.org/ns/1.0'
        ET.register_namespace('tei', TEI_NS)
        
        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as e:
            raise PDFProcessingError(f"Failed to parse TEI XML: {e}")
        
        sections = []
        order = 0
        
        for div in root.findall(f'.//{{{TEI_NS}}}div'):
            # Skip body-level divs
            if div.attrib.get('type') == 'body':
                continue
            
            # Get section name
            section_name = (
                div.attrib.get('type')
                or div.attrib.get('subtype')
                or next(
                    (h.text for h in div.findall(f'./{{{TEI_NS}}}head') if h.text),
                    None
                )
            )
            
            if not section_name:
                continue
            
            # Extract text content
            text_parts = []
            for element in div.iter():
                if element.text and element.tag != f'{{{TEI_NS}}}head':
                    text_parts.append(element.text.strip())
                if element.tail:
                    text_parts.append(element.tail.strip())
            
            content = ' '.join(part for part in text_parts if part)
            
            if content:
                sections.append(PaperSection(
                    name=section_name.lower(),
                    content=content,
                    order=order
                ))
                order += 1
        
        return sections
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "grobid_url": self.settings.grobid_url,
            "grobid_enabled": self.settings.use_grobid,
            "cache_stats": self.cache.get_stats()
        }


# Singleton instance
_paper_service: Optional[PaperService] = None


def get_paper_service() -> PaperService:
    """Get the global paper service instance."""
    global _paper_service
    if _paper_service is None:
        _paper_service = PaperService()
    return _paper_service
```

### backend/services/teaching_service.py
```python
"""
Teaching Service

Main orchestration service that combines paper retrieval, 
lesson generation, and LeetCode functionality.
"""

import time
from typing import Optional, AsyncGenerator

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import PaperNotFoundError
from models.paper import PaperSearchResult, ParsedPaper
from models.lesson import (
    LessonRequest,
    LessonResponse,
    FullLesson,
    StreamingLessonChunk,
)
from models.problem import ProblemRequest, ProblemResponse, LeetCodeProblem
from services.paper_service import get_paper_service
from services.lesson_service import get_lesson_service
from services.leetcode_service import get_leetcode_service
from services.cache_service import get_cache_service

logger = get_logger(__name__)


class TeachingService:
    """
    Main teaching service that orchestrates all functionality.
    
    This is the primary entry point for:
    - Teaching about research topics
    - Generating lessons from papers
    - Providing coding practice
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.paper_service = get_paper_service()
        self.lesson_service = get_lesson_service()
        self.leetcode_service = get_leetcode_service()
        self.cache = get_cache_service()
        
        logger.info("Teaching service initialized")
    
    def teach(self, request: LessonRequest) -> LessonResponse:
        """
        Main teaching endpoint - finds relevant paper and generates lesson.
        
        Args:
            request: Lesson request with query and preferences
            
        Returns:
            Complete lesson response
        """
        start_time = time.time()
        
        try:
            # Search for relevant paper
            logger.info(f"Teaching request: {request.query[:50]}...")
            search_results = self.paper_service.search(request.query, top_k=1)
            
            if not search_results:
                raise PaperNotFoundError(request.query)
            
            best_result = search_results[0]
            logger.info(f"Found paper: {best_result.paper.arxiv_id} (score: {best_result.similarity_score:.2f})")
            
            # Get full paper
            paper = self.paper_service.get_paper(
                str(best_result.paper.url),
                use_grobid=self.settings.use_grobid
            )
            
            # Generate lesson
            lesson = self.lesson_service.generate_lesson(paper, request)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return LessonResponse(
                success=True,
                lesson=lesson,
                cached=False,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Teaching failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return LessonResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    async def teach_streaming(
        self,
        request: LessonRequest
    ) -> AsyncGenerator[StreamingLessonChunk, None]:
        """
        Streaming teaching endpoint - yields chunks as they're generated.
        
        Args:
            request: Lesson request
            
        Yields:
            Streaming lesson chunks
        """
        try:
            # Search for paper
            search_results = self.paper_service.search(request.query, top_k=1)
            
            if not search_results:
                yield StreamingLessonChunk(
                    type="error",
                    data={"message": f"No papers found for: {request.query}"}
                )
                return
            
            best_result = search_results[0]
            
            # Get paper
            paper = self.paper_service.get_paper(
                str(best_result.paper.url),
                use_grobid=self.settings.use_grobid
            )
            
            # Stream lesson generation
            async for chunk in self.lesson_service.generate_lesson_streaming(paper, request):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming teaching failed: {e}")
            yield StreamingLessonChunk(
                type="error",
                data={"message": str(e)}
            )
    
    def get_coding_problem(self, request: Optional[ProblemRequest] = None) -> ProblemResponse:
        """
        Get a coding problem for practice.
        
        Args:
            request: Problem request criteria
            
        Returns:
            Problem response
        """
        start_time = time.time()
        
        try:
            problem = self.leetcode_service.get_random_problem(request)
            processing_time = int((time.time() - start_time) * 1000)
            
            return ProblemResponse(
                success=True,
                problem=problem,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to get coding problem: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            return ProblemResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    def search_papers(self, query: str, top_k: int = 5) -> list[PaperSearchResult]:
        """
        Search for papers without generating lessons.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of search results
        """
        return self.paper_service.search(query, top_k=top_k)
    
    def get_paper_details(self, url: str) -> ParsedPaper:
        """
        Get full paper details.
        
        Args:
            url: Paper URL
            
        Returns:
            Parsed paper
        """
        return self.paper_service.get_paper(url)
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "paper_service": self.paper_service.get_stats(),
            "leetcode_service": self.leetcode_service.get_stats(),
            "cache": self.cache.get_stats()
        }


# Singleton instance
_teaching_service: Optional[TeachingService] = None


def get_teaching_service() -> TeachingService:
    """Get the global teaching service instance."""
    global _teaching_service
    if _teaching_service is None:
        _teaching_service = TeachingService()
    return _teaching_service
```

### backend/api/routes/__init__.py
```python
"""
API Routes Package
"""

from . import health
from . import teach
from . import leetcode

__all__ = ["health", "teach", "leetcode"]
```

### backend/api/routes/health.py
```python
"""
Health Check Routes

Endpoints for monitoring application health.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from core.config import get_settings
from services.teaching_service import get_teaching_service

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    service: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check with service stats."""
    status: str
    version: str
    service: str
    stats: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        service=settings.app_name
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with service statistics.
    
    Returns:
        Detailed health status with stats
    """
    settings = get_settings()
    
    try:
        teaching_service = get_teaching_service()
        stats = teaching_service.get_stats()
        status = "healthy"
    except Exception as e:
        stats = {"error": str(e)}
        status = "degraded"
    
    return DetailedHealthResponse(
        status=status,
        version=settings.app_version,
        service=settings.app_name,
        stats=stats
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
```

### backend/api/routes/leetcode.py
```python
"""
LeetCode Routes

Endpoints for coding practice.
"""

from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from core.logging import get_logger
from models.problem import ProblemRequest, ProblemResponse, ProblemDifficulty
from services.teaching_service import get_teaching_service

router = APIRouter()
logger = get_logger(__name__)


class CodingProblemRequest(BaseModel):
    """Request for a coding problem."""
    difficulties: List[ProblemDifficulty] = Field(
        default=[ProblemDifficulty.MEDIUM, ProblemDifficulty.HARD],
        description="Allowed difficulty levels"
    )
    exclude_premium: bool = Field(True, description="Exclude premium problems")
    
    class Config:
        json_schema_extra = {
            "example": {
                "difficulties": ["Medium", "Hard"],
                "exclude_premium": True
            }
        }


@router.post("/leetcode/random", response_model=ProblemResponse)
async def get_random_problem(request: Optional[CodingProblemRequest] = None):
    """
    Get a random LeetCode problem for practice.
    
    By default, returns Medium or Hard problems (non-premium).
    
    Returns:
        Random problem with statement and metadata
    """
    request = request or CodingProblemRequest()
    logger.info(f"LeetCode request: difficulties={request.difficulties}")
    
    teaching_service = get_teaching_service()
    
    problem_request = ProblemRequest(
        difficulties=request.difficulties,
        exclude_premium=request.exclude_premium
    )
    
    return teaching_service.get_coding_problem(problem_request)


@router.get("/leetcode/problem/{slug}", response_model=ProblemResponse)
async def get_problem_by_slug(slug: str):
    """
    Get a specific LeetCode problem by slug.
    
    Args:
        slug: Problem URL slug (e.g., "two-sum")
        
    Returns:
        The requested problem
    """
    logger.info(f"LeetCode slug request: {slug}")
    
    teaching_service = get_teaching_service()
    
    try:
        problem = teaching_service.leetcode_service.get_problem_by_slug(slug)
        return ProblemResponse(success=True, problem=problem)
    except Exception as e:
        return ProblemResponse(success=False, error=str(e))
```

### backend/api/routes/teach.py
```python
"""
Teaching Routes

Endpoints for lesson generation and paper search.
"""

import json
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.logging import get_logger
from models.lesson import LessonRequest, LessonResponse, LessonDifficulty
from models.paper import PaperSearchRequest
from services.teaching_service import get_teaching_service

router = APIRouter()
logger = get_logger(__name__)


class TeachRequest(BaseModel):
    """Request to learn about a topic."""
    query: str = Field(..., min_length=3, max_length=500, description="What do you want to learn?")
    difficulty: LessonDifficulty = Field(LessonDifficulty.BEGINNER, description="Lesson difficulty")
    include_examples: bool = Field(True, description="Include examples")
    include_math: bool = Field(True, description="Include step-by-step math")
    max_sections: Optional[int] = Field(None, ge=1, le=20, description="Limit sections")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "attention mechanisms in transformers",
                "difficulty": "beginner",
                "include_examples": True,
                "include_math": True
            }
        }


class SearchRequest(BaseModel):
    """Request to search for papers."""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


@router.post("/teach", response_model=LessonResponse)
async def teach(request: TeachRequest):
    """
    Generate a lesson about a topic.
    
    This endpoint:
    1. Searches for the most relevant research paper
    2. Parses the paper into sections
    3. Generates beginner-friendly explanations for each section
    
    Returns:
        Complete lesson with all sections
    """
    logger.info(f"Teach request: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    
    lesson_request = LessonRequest(
        query=request.query,
        difficulty=request.difficulty,
        include_examples=request.include_examples,
        include_math=request.include_math,
        max_sections=request.max_sections
    )
    
    return teaching_service.teach(lesson_request)


@router.post("/teach/stream")
async def teach_streaming(request: TeachRequest):
    """
    Generate a lesson with streaming responses.
    
    Returns Server-Sent Events (SSE) as sections are generated.
    
    Event types:
    - `metadata`: Paper information
    - `section`: Generated lesson section
    - `done`: Generation complete
    - `error`: Error occurred
    """
    logger.info(f"Streaming teach request: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    
    lesson_request = LessonRequest(
        query=request.query,
        difficulty=request.difficulty,
        include_examples=request.include_examples,
        include_math=request.include_math,
        max_sections=request.max_sections
    )
    
    async def event_generator():
        async for chunk in teaching_service.teach_streaming(lesson_request):
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/search")
async def search_papers(request: SearchRequest):
    """
    Search for relevant papers.
    
    Returns a list of papers matching the query, ranked by relevance.
    """
    logger.info(f"Search request: {request.query[:50]}...")
    
    teaching_service = get_teaching_service()
    results = teaching_service.search_papers(request.query, top_k=request.top_k)
    
    return {
        "query": request.query,
        "results": [r.model_dump() for r in results],
        "count": len(results)
    }


@router.get("/paper")
async def get_paper(
    url: str = Query(..., description="arXiv paper URL")
):
    """
    Get detailed information about a specific paper.
    
    Returns parsed paper with sections.
    """
    logger.info(f"Paper request: {url}")
    
    teaching_service = get_teaching_service()
    paper = teaching_service.get_paper_details(url)
    
    return paper.model_dump()
```

### backend/models/__init__.py
```python
"""
Data Models Package

Pydantic models for all data structures.
"""

from .paper import (
    PaperMetadata,
    PaperSection,
    ParsedPaper,
    PaperSearchResult,
    PaperSearchRequest,
)
from .lesson import (
    LessonDifficulty,
    LessonFragment,
    FullLesson,
    LessonRequest,
    LessonResponse,
    StreamingLessonChunk,
)
from .problem import (
    ProblemDifficulty,
    LeetCodeProblem,
    ProblemRequest,
    ProblemResponse,
    ProblemCatalogEntry,
)

__all__ = [
    # Paper models
    "PaperMetadata",
    "PaperSection",
    "ParsedPaper",
    "PaperSearchResult",
    "PaperSearchRequest",
    # Lesson models
    "LessonDifficulty",
    "LessonFragment",
    "FullLesson",
    "LessonRequest",
    "LessonResponse",
    "StreamingLessonChunk",
    # Problem models
    "ProblemDifficulty",
    "LeetCodeProblem",
    "ProblemRequest",
    "ProblemResponse",
    "ProblemCatalogEntry",
]
```

### backend/models/lesson.py
```python
"""
Lesson Data Models

Pydantic models for lesson-related data structures.
"""

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class LessonDifficulty(str, Enum):
    """Lesson difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class LessonFragment(BaseModel):
    """A single lesson fragment from a paper section."""
    
    section_name: str = Field(..., description="Original section name")
    content: str = Field(..., description="Beginner-friendly lesson content")
    order: int = Field(..., description="Order in the full lesson")
    has_math: bool = Field(False, description="Whether section includes math")
    has_code: bool = Field(False, description="Whether section includes code")
    estimated_read_time: int = Field(0, description="Estimated read time in minutes")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.estimated_read_time == 0:
            # Average reading speed: 200 words per minute
            word_count = len(self.content.split())
            self.estimated_read_time = max(1, word_count // 200)


class FullLesson(BaseModel):
    """A complete lesson generated from a paper."""
    
    paper_id: str = Field(..., description="Source paper arXiv ID")
    paper_title: str = Field(..., description="Source paper title")
    paper_url: str = Field(..., description="Source paper URL")
    query: str = Field(..., description="Original user query")
    
    fragments: list[LessonFragment] = Field(default_factory=list)
    
    difficulty: LessonDifficulty = Field(LessonDifficulty.BEGINNER)
    total_read_time: int = Field(0, description="Total estimated read time")
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time_seconds: float = Field(0, description="Time to generate")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total_read_time == 0 and self.fragments:
            self.total_read_time = sum(f.estimated_read_time for f in self.fragments)
    
    @property
    def full_content(self) -> str:
        """Get the full lesson as a single string."""
        parts = []
        for fragment in sorted(self.fragments, key=lambda f: f.order):
            parts.append(f"## {fragment.section_name.title()}\n\n{fragment.content}")
        return "\n\n---\n\n".join(parts)
    
    @property
    def table_of_contents(self) -> list[str]:
        """Get section names as table of contents."""
        return [f.section_name.title() for f in sorted(self.fragments, key=lambda f: f.order)]


class LessonRequest(BaseModel):
    """Request to generate a lesson."""
    
    query: str = Field(..., min_length=3, max_length=500, description="What to learn about")
    difficulty: LessonDifficulty = Field(LessonDifficulty.BEGINNER)
    include_examples: bool = Field(True, description="Include examples in explanations")
    include_math: bool = Field(True, description="Include step-by-step math")
    max_sections: Optional[int] = Field(None, ge=1, le=20, description="Limit sections")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "attention mechanisms in transformers",
                "difficulty": "beginner",
                "include_examples": True,
                "include_math": True
            }
        }


class LessonResponse(BaseModel):
    """Response containing a generated lesson."""
    
    success: bool = Field(True)
    lesson: Optional[FullLesson] = None
    error: Optional[str] = None
    
    # Metadata
    cached: bool = Field(False, description="Whether result was from cache")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")


class StreamingLessonChunk(BaseModel):
    """A chunk of a streaming lesson response."""
    
    type: str = Field(..., description="Chunk type: 'metadata', 'section', 'done', 'error'")
    data: dict = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "metadata", "data": {"paper_title": "Attention Is All You Need", "total_sections": 5}},
                {"type": "section", "data": {"name": "Introduction", "content": "Let's start..."}},
                {"type": "done", "data": {"total_time_seconds": 45.2}},
                {"type": "error", "data": {"message": "Failed to process"}}
            ]
        }
```

### backend/models/paper.py
```python
"""
Paper Data Models

Pydantic models for paper-related data structures.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PaperMetadata(BaseModel):
    """Metadata for a research paper."""
    
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    url: HttpUrl = Field(..., description="Paper URL")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    authors: list[str] = Field(default_factory=list, description="Paper authors")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "abstract": "The dominant sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "categories": ["cs.CL", "cs.LG"],
                "published_date": "2017-06-12T00:00:00Z"
            }
        }


class PaperSection(BaseModel):
    """A section extracted from a paper."""
    
    name: str = Field(..., description="Section name/title")
    content: str = Field(..., description="Section text content")
    order: int = Field(..., description="Section order in paper")
    word_count: int = Field(0, description="Word count")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class ParsedPaper(BaseModel):
    """A fully parsed paper with sections."""
    
    metadata: PaperMetadata
    sections: list[PaperSection] = Field(default_factory=list)
    raw_text: Optional[str] = Field(None, description="Full raw text")
    parsing_method: str = Field("grobid", description="Method used to parse (grobid/abstract)")
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def section_names(self) -> list[str]:
        return [s.name for s in self.sections]
    
    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)


class PaperSearchResult(BaseModel):
    """Result from a paper search."""
    
    paper: PaperMetadata
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    index_position: int = Field(..., description="Position in FAISS index")
    
    class Config:
        json_schema_extra = {
            "example": {
                "paper": {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762"
                },
                "similarity_score": 0.92,
                "index_position": 42
            }
        }


class PaperSearchRequest(BaseModel):
    """Request to search for papers."""
    
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    top_k: int = Field(1, ge=1, le=10, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "attention mechanisms in transformers",
                "top_k": 3
            }
        }
```

### backend/models/problem.py
```python
"""
LeetCode Problem Models

Pydantic models for LeetCode-related data structures.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class ProblemDifficulty(str, Enum):
    """LeetCode problem difficulty levels."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class LeetCodeProblem(BaseModel):
    """A LeetCode problem."""
    
    title: str = Field(..., description="Problem title")
    slug: str = Field(..., description="URL slug")
    difficulty: ProblemDifficulty = Field(..., description="Difficulty level")
    statement: str = Field(..., description="Problem statement")
    url: str = Field("", description="Full LeetCode URL")
    
    # Optional metadata
    acceptance_rate: Optional[float] = Field(None, description="Acceptance rate percentage")
    topics: list[str] = Field(default_factory=list, description="Related topics")
    hints: list[str] = Field(default_factory=list, description="Problem hints")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.url and self.slug:
            self.url = f"https://leetcode.com/problems/{self.slug}/"
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Two Sum",
                "slug": "two-sum",
                "difficulty": "Easy",
                "statement": "Given an array of integers nums and an integer target...",
                "url": "https://leetcode.com/problems/two-sum/",
                "topics": ["Array", "Hash Table"]
            }
        }


class ProblemRequest(BaseModel):
    """Request for a LeetCode problem."""
    
    difficulties: list[ProblemDifficulty] = Field(
        default=[ProblemDifficulty.MEDIUM, ProblemDifficulty.HARD],
        description="Allowed difficulties"
    )
    topics: Optional[list[str]] = Field(None, description="Filter by topics")
    exclude_premium: bool = Field(True, description="Exclude premium problems")
    
    class Config:
        json_schema_extra = {
            "example": {
                "difficulties": ["Medium", "Hard"],
                "exclude_premium": True
            }
        }


class ProblemResponse(BaseModel):
    """Response containing a LeetCode problem."""
    
    success: bool = Field(True)
    problem: Optional[LeetCodeProblem] = None
    error: Optional[str] = None
    
    # Metadata
    cached: bool = Field(False)
    processing_time_ms: int = Field(0)


class ProblemCatalogEntry(BaseModel):
    """Entry in the LeetCode problem catalog."""
    
    slug: str
    title: str
    difficulty: ProblemDifficulty
    paid_only: bool = False
    acceptance_rate: Optional[float] = None
```

### backend/api/main.py
```python
"""
FastAPI Application

Main entry point for the LLM Teaching Assistant API.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger
from core.exceptions import BaseAppException
from api.routes import teach, leetcode, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        🎓 **LLM Teaching Assistant API**
        
        An AI-powered teaching assistant that:
        - Retrieves and explains research papers from arXiv
        - Converts academic content into beginner-friendly lessons
        - Provides LeetCode problems for coding practice
        
        ## Features
        
        - **Semantic Search**: Find relevant papers using natural language
        - **Lesson Generation**: Get step-by-step explanations
        - **Streaming Support**: Real-time lesson generation via SSE
        - **Coding Practice**: Random LeetCode problems
        
        ## Quick Start
        
        ```python
        import requests
        
        # Generate a lesson
        response = requests.post(
            "http://localhost:8000/api/v1/teach",
            json={"query": "attention mechanisms in transformers"}
        )
        lesson = response.json()
        ```
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Exception handler
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        logger.error(f"Application error: {exc.code} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    # Generic exception handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred"
                }
            }
        )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(teach.router, prefix=settings.api_prefix, tags=["Teaching"])
    app.include_router(leetcode.router, prefix=settings.api_prefix, tags=["LeetCode"])
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
```

## 3. Frontend Code

### frontend/src/App.tsx
```tsx
import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { ThemeProvider } from '@/hooks/useTheme'
import { Header, Hero, LessonDisplay, ProblemDisplay, LoadingOverlay } from '@/components'
import { generateLesson, getRandomProblem, Lesson, Problem, LessonRequest } from '@/lib/api'

type ViewState = 
  | { type: 'home' }
  | { type: 'loading'; message: string }
  | { type: 'lesson'; lesson: Lesson }
  | { type: 'problem'; problem: Problem }
  | { type: 'error'; message: string }

export default function App() {
  const [viewState, setViewState] = useState<ViewState>({ type: 'home' })
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (query: string, mode: 'learn' | 'code') => {
    setIsLoading(true)

    try {
      if (mode === 'learn') {
        setViewState({ type: 'loading', message: 'Searching for relevant papers...' })
        
        const request: LessonRequest = {
          query,
          difficulty: 'beginner',
          include_examples: true,
          include_math: true,
          max_sections: 5,
        }

        const response = await generateLesson(request)

        if (response.success && response.lesson) {
          setViewState({ type: 'lesson', lesson: response.lesson })
        } else {
          setViewState({ 
            type: 'error', 
            message: response.error || 'Failed to generate lesson' 
          })
        }
      } else {
        setViewState({ type: 'loading', message: 'Finding a coding challenge...' })
        
        const response = await getRandomProblem()

        if (response.success && response.problem) {
          setViewState({ type: 'problem', problem: response.problem })
        } else {
          setViewState({ 
            type: 'error', 
            message: response.error || 'Failed to fetch problem' 
          })
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setViewState({ 
        type: 'error', 
        message: 'Something went wrong. Please try again.' 
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleClose = () => {
    setViewState({ type: 'home' })
  }

  const handleNewProblem = async () => {
    setIsLoading(true)
    try {
      const response = await getRandomProblem()
      if (response.success && response.problem) {
        setViewState({ type: 'problem', problem: response.problem })
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 transition-colors">
        <Header />
        
        <main>
          <Hero onSubmit={handleSubmit} isLoading={isLoading} />
        </main>

        <AnimatePresence>
          {viewState.type === 'loading' && (
            <LoadingOverlay message={viewState.message} />
          )}

          {viewState.type === 'lesson' && (
            <LessonDisplay 
              lesson={viewState.lesson} 
              onClose={handleClose} 
            />
          )}

          {viewState.type === 'problem' && (
            <ProblemDisplay
              problem={viewState.problem}
              onClose={handleClose}
              onNewProblem={handleNewProblem}
              isLoading={isLoading}
            />
          )}
        </AnimatePresence>

        <AnimatePresence>
          {viewState.type === 'error' && (
            <div className="fixed bottom-4 right-4 z-50">
              <div className="bg-red-500 text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3">
                <span>{viewState.message}</span>
                <button
                  onClick={handleClose}
                  className="text-white/80 hover:text-white"
                >
                  ✕
                </button>
              </div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </ThemeProvider>
  )
}
```

### frontend/src/lib/api.ts
```typescript
const API_BASE = (import.meta as any).env?.VITE_API_URL 
  ? `${(import.meta as any).env.VITE_API_URL}/api/v1`
  : '/api/v1'

export interface LessonRequest {
  query: string
  difficulty?: 'beginner' | 'intermediate' | 'advanced'
  include_examples?: boolean
  include_math?: boolean
  max_sections?: number
}

export interface LessonFragment {
  section_name: string
  content: string
  order: number
  estimated_read_time: number
}

export interface Lesson {
  paper_id: string
  paper_title: string
  paper_url: string
  query: string
  fragments: LessonFragment[]
  total_read_time: number
  generation_time_seconds: number
}

export interface LessonResponse {
  success: boolean
  lesson?: Lesson
  error?: string
  processing_time_ms: number
}

export interface Problem {
  title: string
  slug: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  statement: string
  url: string
  topics: string[]
}

export interface ProblemResponse {
  success: boolean
  problem?: Problem
  error?: string
  processing_time_ms: number
}

export interface StreamChunk {
  type: 'metadata' | 'section' | 'done' | 'error'
  data: Record<string, unknown>
}

export async function generateLesson(request: LessonRequest): Promise<LessonResponse> {
  const response = await fetch(`${API_BASE}/teach`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return response.json()
}

export async function getRandomProblem(
  difficulties: string[] = ['Medium', 'Hard']
): Promise<ProblemResponse> {
  const response = await fetch(`${API_BASE}/leetcode/random`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ difficulties, exclude_premium: true }),
  })
  return response.json()
}

export async function searchPapers(query: string, topK: number = 5) {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  return response.json()
}
```

### frontend/src/components/Button.tsx
```tsx
import { forwardRef, ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', isLoading, children, disabled, ...props }, ref) => {
    const baseStyles = 'inline-flex items-center justify-center font-medium rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed'
    
    const variants = {
      primary: 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white shadow-lg shadow-primary-500/25 hover:shadow-xl hover:shadow-primary-500/30 focus:ring-primary-500',
      secondary: 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-gray-500',
      ghost: 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 focus:ring-gray-500',
      outline: 'border-2 border-primary-500 text-primary-500 hover:bg-primary-50 dark:hover:bg-primary-950 focus:ring-primary-500',
    }
    
    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-sm',
      lg: 'px-6 py-3 text-base',
    }

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading && (
          <svg
            className="animate-spin -ml-1 mr-2 h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'

export default Button
```

### frontend/src/components/Card.tsx
```tsx
import { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface CardProps {
  children: ReactNode
  className?: string
  hover?: boolean
  glass?: boolean
}

export function Card({ children, className, hover = false, glass = false }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border',
        glass 
          ? 'glass glass-border' 
          : 'bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800',
        hover && 'transition-all duration-300 hover:shadow-xl hover:shadow-primary-500/5 hover:-translate-y-1',
        className
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4 border-b border-gray-200 dark:border-gray-800', className)}>
      {children}
    </div>
  )
}

export function CardContent({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('px-6 py-4', className)}>
      {children}
    </div>
  )
}
```

### frontend/src/components/Header.tsx
```tsx
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sun, Moon, Menu, X, GraduationCap } from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'

interface NavItem {
  label: string
  href: string
}

const navItems: NavItem[] = [
  { label: 'Learn', href: '#learn' },
  { label: 'Practice', href: '#practice' },
  { label: 'Search', href: '#search' },
]

export default function Header() {
  const { setTheme, resolvedTheme } = useTheme()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const toggleTheme = () => {
    setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div className="glass glass-border mx-4 mt-4 rounded-2xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <motion.a
              href="/"
              className="flex items-center space-x-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-white" />
              </div>
              <span className="font-bold text-xl text-gradient hidden sm:block">
                LearnAI
              </span>
            </motion.a>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-1">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="px-4 py-2 rounded-lg text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  {item.label}
                </a>
              ))}
            </nav>

            {/* Right side */}
            <div className="flex items-center space-x-2">
              {/* Theme toggle */}
              <motion.button
                onClick={toggleTheme}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <AnimatePresence mode="wait">
                  {resolvedTheme === 'dark' ? (
                    <motion.div
                      key="sun"
                      initial={{ rotate: -90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: 90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Sun className="w-5 h-5 text-yellow-500" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="moon"
                      initial={{ rotate: 90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: -90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Moon className="w-5 h-5 text-gray-600" />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>

              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
              >
                {mobileMenuOpen ? (
                  <X className="w-5 h-5" />
                ) : (
                  <Menu className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-gray-200 dark:border-gray-700"
            >
              <div className="px-4 py-3 space-y-1">
                {navItems.map((item) => (
                  <a
                    key={item.label}
                    href={item.href}
                    className="block px-4 py-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    {item.label}
                  </a>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  )
}
```

### frontend/src/components/Hero.tsx
```tsx
import { useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, ArrowRight, BookOpen, Code } from 'lucide-react'
import Button from './Button'
import { Textarea } from './Input'

interface HeroProps {
  onSubmit: (query: string, mode: 'learn' | 'code') => void
  isLoading: boolean
}

export default function Hero({ onSubmit, isLoading }: HeroProps) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'learn' | 'code'>('learn')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit(query, mode)
    }
  }

  const suggestions = [
    'Explain attention mechanisms in transformers',
    'How does BERT pre-training work?',
    'What is LoRA fine-tuning?',
    'Explain the GPT architecture',
  ]

  return (
    <section className="relative min-h-screen flex items-center justify-center px-4 pt-24 pb-12">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 dark:bg-primary-950 text-primary-600 dark:text-primary-400 text-sm font-medium mb-6"
        >
          <Sparkles className="w-4 h-4" />
          <span>AI-Powered Learning</span>
        </motion.div>

        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-6"
        >
          Learn AI Research
          <br />
          <span className="text-gradient">The Easy Way</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-lg sm:text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto"
        >
          Transform complex research papers into beginner-friendly lessons.
          Practice coding with curated LeetCode problems.
        </motion.p>

        {/* Mode switcher */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex justify-center gap-2 mb-6"
        >
          <button
            onClick={() => setMode('learn')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
              mode === 'learn'
                ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Learn
          </button>
          <button
            onClick={() => setMode('code')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
              mode === 'code'
                ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <Code className="w-4 h-4" />
            Practice
          </button>
        </motion.div>

        {/* Input form */}
        <motion.form
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          onSubmit={handleSubmit}
          className="relative max-w-2xl mx-auto"
        >
          <div className="relative">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                mode === 'learn'
                  ? 'What would you like to learn about? (e.g., "Explain transformers")'
                  : 'Describe what you want to practice...'
              }
              rows={3}
              className="pr-24 text-lg"
            />
            <div className="absolute right-2 bottom-2">
              <Button
                type="submit"
                isLoading={isLoading}
                disabled={!query.trim()}
                className="rounded-xl"
              >
                {isLoading ? (
                  'Generating...'
                ) : (
                  <>
                    Go
                    <ArrowRight className="w-4 h-4 ml-1" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </motion.form>

        {/* Suggestions */}
        {mode === 'learn' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="mt-6 flex flex-wrap justify-center gap-2"
          >
            <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">
              Try:
            </span>
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setQuery(suggestion)}
                className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </motion.div>
        )}
      </div>
    </section>
  )
}
```

### frontend/src/components/Input.tsx
```tsx
import { forwardRef, InputHTMLAttributes, TextareaHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          'w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700',
          'bg-white dark:bg-gray-800',
          'text-gray-900 dark:text-gray-100',
          'placeholder:text-gray-400 dark:placeholder:text-gray-500',
          'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
          'transition-all duration-200',
          className
        )}
        {...props}
      />
    )
  }
)

Input.displayName = 'Input'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          'w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700',
          'bg-white dark:bg-gray-800',
          'text-gray-900 dark:text-gray-100',
          'placeholder:text-gray-400 dark:placeholder:text-gray-500',
          'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
          'transition-all duration-200 resize-none',
          className
        )}
        {...props}
      />
    )
  }
)

Textarea.displayName = 'Textarea'
```

### frontend/src/components/LessonDisplay.tsx
```tsx
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { 
  BookOpen, 
  Clock, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp,
  Copy,
  Check,
  X
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import Button from './Button'
import { Lesson, LessonFragment } from '@/lib/api'
import { formatReadTime } from '@/lib/utils'

interface LessonDisplayProps {
  lesson: Lesson
  onClose: () => void
}

export default function LessonDisplay({ lesson, onClose }: LessonDisplayProps) {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(
    new Set(lesson.fragments.map((_, i) => i))
  )
  const [copied, setCopied] = useState(false)

  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSections(newExpanded)
  }

  const copyLesson = async () => {
    const fullContent = lesson.fragments
      .map((f) => `## ${f.section_name}\n\n${f.content}`)
      .join('\n\n---\n\n')
    
    await navigator.clipboard.writeText(fullContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm"
    >
      <div className="min-h-screen px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="max-w-4xl mx-auto"
        >
          <Card className="shadow-2xl">
            {/* Header */}
            <CardHeader className="relative">
              <div className="flex items-start justify-between">
                <div className="flex-1 pr-8">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <BookOpen className="w-4 h-4" />
                    <span>Lesson from research paper</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    {lesson.paper_title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      <span>{formatReadTime(lesson.total_read_time)}</span>
                    </div>
                    <a
                      href={lesson.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 hover:text-primary-500 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>View Paper</span>
                    </a>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyLesson}
                    className="text-gray-500"
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-gray-500"
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>
              </div>
            </CardHeader>

            {/* Table of Contents */}
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">
                Table of Contents
              </h3>
              <div className="flex flex-wrap gap-2">
                {lesson.fragments.map((fragment, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setExpandedSections(new Set([index]))
                      document.getElementById(`section-${index}`)?.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start',
                      })
                    }}
                    className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {fragment.section_name}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <CardContent className="space-y-4">
              {lesson.fragments.map((fragment, index) => (
                <LessonSection
                  key={index}
                  fragment={fragment}
                  index={index}
                  isExpanded={expandedSections.has(index)}
                  onToggle={() => toggleSection(index)}
                />
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}

interface LessonSectionProps {
  fragment: LessonFragment
  index: number
  isExpanded: boolean
  onToggle: () => void
}

function LessonSection({ fragment, index, isExpanded, onToggle }: LessonSectionProps) {
  return (
    <motion.div
      id={`section-${index}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden"
    >
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="w-6 h-6 rounded-full bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 text-sm font-medium flex items-center justify-center">
            {index + 1}
          </span>
          <h3 className="font-semibold text-gray-900 dark:text-white capitalize">
            {fragment.section_name}
          </h3>
        </div>
        <div className="flex items-center gap-2 text-gray-500">
          <span className="text-sm">{formatReadTime(fragment.estimated_read_time)}</span>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5" />
          ) : (
            <ChevronDown className="w-5 h-5" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-4 prose-custom">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {fragment.content}
              </ReactMarkdown>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
```

### frontend/src/components/Loading.tsx
```tsx
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  }

  return (
    <div className={cn('relative', sizes[size], className)}>
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-primary-200 dark:border-primary-800"
      />
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary-500"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      />
    </div>
  )
}

interface LoadingOverlayProps {
  message?: string
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm"
    >
      <div className="text-center">
        <LoadingSpinner size="lg" className="mx-auto mb-4" />
        <p className="text-gray-600 dark:text-gray-400">{message}</p>
      </div>
    </motion.div>
  )
}
```

### frontend/src/components/ProblemDisplay.tsx
```tsx
import { motion } from 'framer-motion'
import { 
  Code, 
  ExternalLink, 
  X,
  Tag,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader } from './Card'
import Button from './Button'
import { Problem } from '@/lib/api'
import { cn } from '@/lib/utils'

interface ProblemDisplayProps {
  problem: Problem
  onClose: () => void
  onNewProblem: () => void
  isLoading: boolean
}

export default function ProblemDisplay({ 
  problem, 
  onClose, 
  onNewProblem,
  isLoading 
}: ProblemDisplayProps) {
  const difficultyColors: Record<string, string> = {
    Easy: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    Medium: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
    Hard: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300',
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm"
    >
      <div className="min-h-screen px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 50, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="max-w-4xl mx-auto"
        >
          <Card className="shadow-2xl">
            {/* Header */}
            <CardHeader className="relative">
              <div className="flex items-start justify-between">
                <div className="flex-1 pr-8">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-2">
                    <Code className="w-4 h-4" />
                    <span>Coding Challenge</span>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
                    {problem.title}
                  </h2>
                  <div className="flex flex-wrap items-center gap-3">
                    <span className={cn(
                      'px-3 py-1 rounded-full text-sm font-medium',
                      difficultyColors[problem.difficulty] || difficultyColors['Medium']
                    )}>
                      {problem.difficulty}
                    </span>
                    <a
                      href={problem.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-sm text-primary-500 hover:text-primary-600 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                      <span>Solve on LeetCode</span>
                    </a>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onNewProblem}
                    isLoading={isLoading}
                  >
                    <RefreshCw className="w-4 h-4 mr-1" />
                    New Problem
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-gray-500"
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>
              </div>
            </CardHeader>

            {/* Topics */}
            {problem.topics && problem.topics.length > 0 && (
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
                <div className="flex items-center gap-2 mb-2">
                  <Tag className="w-4 h-4 text-gray-500" />
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Related Topics
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {problem.topics.map((topic) => (
                    <span
                      key={topic}
                      className="text-sm px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Problem Statement */}
            <CardContent>
              <div className="prose-custom">
                <h3 className="text-lg font-semibold mb-4">Problem Statement</h3>
                <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl p-4 font-mono text-sm whitespace-pre-wrap">
                  {problem.statement}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}
```

## 4. Dependencies

### backend/requirements.txt
```
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# AI/ML
openai>=1.10.0
faiss-cpu>=1.7.4
numpy>=1.24.0

# HTTP
requests>=2.31.0
httpx>=0.26.0
aiohttp>=3.9.0

# Parsing
beautifulsoup4>=4.12.0
lxml>=5.1.0

# Utilities
python-dotenv>=1.0.0
python-multipart>=0.0.6

# Development
pytest>=7.4.0
pytest-asyncio>=0.23.0
black>=24.0.0
isort>=5.13.0
mypy>=1.8.0

# Optional: LangGraph (if you want to keep agent functionality)
# langgraph>=0.0.20
# langchain>=0.1.0
# langmem>=0.0.30
```

### frontend/package.json
```json
{
  "name": "llm-teaching-assistant-ui",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0",
    "framer-motion": "^10.18.0",
    "lucide-react": "^0.303.0",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.45",
    "@types/react-dom": "^18.2.18",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.3.3",
    "vite": "^5.0.10"
  }
}
```

## 5. Environment Variables

### backend/.env (sanitized)
```
# LLM Teaching Assistant Configuration
# Copy this file to .env and fill in your values

# =<REDACTED>
# REQUIRED
# =<REDACTED>

# OpenAI API Key (required)
OPENAI_API_KEY=<REDACTED>

# =<REDACTED>
# OPTIONAL - API Settings
# =<REDACTED>

# API host and port
API_HOST=<REDACTED>
API_PORT=<REDACTED>

# Debug mode (set to true for development)
DEBUG=<REDACTED>

# =<REDACTED>
# OPTIONAL - Model Settings
# =<REDACTED>

# OpenAI models
EMBEDDING_MODEL=<REDACTED>
CHAT_MODEL=<REDACTED>
LESSON_MODEL=<REDACTED>

# =<REDACTED>
# OPTIONAL - GROBID Settings
# =<REDACTED>

# GROBID service URL (for PDF parsing)
# Use the cloud service or run locally
GROBID_URL=<REDACTED>

# GROBID timeout in seconds
GROBID_TIMEOUT=<REDACTED>

# Set to false to use abstract-only mode (no PDF parsing)
USE_GROBID=<REDACTED>

# =<REDACTED>
# OPTIONAL - File Paths
# =<REDACTED>

# Data directory
DATA_DIR=<REDACTED>

# FAISS index path
FAISS_INDEX_PATH=<REDACTED>

# URLs JSON path
URLS_JSON_PATH=<REDACTED>

# Cache directory
CACHE_DIR=<REDACTED>

# =<REDACTED>
# OPTIONAL - Cache Settings
# =<REDACTED>

# Enable/disable caching
CACHE_ENABLED=<REDACTED>

# Cache TTL in seconds (default: 24 hours)
CACHE_TTL=<REDACTED>

# =<REDACTED>
# OPTIONAL - Rate Limiting
# =<REDACTED>

# Enable rate limiting
RATE_LIMIT_ENABLED=<REDACTED>

# Requests per window
RATE_LIMIT_REQUESTS=<REDACTED>

# Window size in seconds
RATE_LIMIT_WINDOW=<REDACTED>

# =<REDACTED>
# OPTIONAL - Logging
# =<REDACTED>

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=<REDACTED>

# Log format (json or text)
LOG_FORMAT=<REDACTED>
```

---
## What I Need Claude To Do

Implement v2 improvements:
1. Add relevance threshold system (0.50/0.35 cutoffs)
2. Add Semantic Scholar integration for dynamic paper fetching
3. Add Pinecone service (optional, for production)
4. Add Query Enhancement service (intent detection)
5. Remove LeetCode feature
6. Update teaching_service.py with new orchestration logic
