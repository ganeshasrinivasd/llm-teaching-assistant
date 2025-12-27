"""
Services Package v2

Business logic layer for the application.

Changes in v2:
- Removed LeetCodeService
- Added QueryService (query enhancement)
- Added ScholarService (dynamic paper fetching)
"""

from .cache_service import CacheService, get_cache_service
from .embedding_service import EmbeddingService, get_embedding_service
from .paper_service import PaperService, get_paper_service
from .lesson_service import LessonService, get_lesson_service
from .teaching_service import TeachingService, get_teaching_service
from .query_service import QueryService, get_query_service
from .scholar_service import ScholarService, get_scholar_service

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
    # Teaching (main orchestrator)
    "TeachingService",
    "get_teaching_service",
    # Query Enhancement (NEW)
    "QueryService",
    "get_query_service",
    # Scholar Service (NEW)
    "ScholarService",
    "get_scholar_service",
]
