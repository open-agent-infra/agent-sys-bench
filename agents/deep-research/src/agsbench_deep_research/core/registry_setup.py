"""
Centralized dependency registry setup.
This module initializes all service registrations in one place.
"""

from .di import dependency_registry
from .service.llm import LLMService
from .service.vdb import VectorDBService
from .service.embedding import EmbeddingService
from .service.search import SearchService
from .service.caching import CachingService
from .services_manager import BaseServiceManager
from .config import ConfigurationManager, config_manager
from .factories import (
    create_llm_service,
    create_vdb_service,
    create_service_manager,
    create_embedding_service,
    create_search_service,
    create_caching_service,  # Register caching factory
)
import logging

logger = logging.getLogger(__name__)


def setup_registry():
    """Set up all service registrations."""
    logger.info("Setting up dependency registry...")
    try:
        # Register configuration manager
        dependency_registry.register(ConfigurationManager, config_manager)

        # Register core services
        dependency_registry.register_factory(LLMService, create_llm_service)
        dependency_registry.register_factory(VectorDBService, create_vdb_service)
        dependency_registry.register_factory(EmbeddingService, create_embedding_service)
        dependency_registry.register_factory(SearchService, create_search_service)
        dependency_registry.register_factory(BaseServiceManager, create_service_manager)
        dependency_registry.register_factory(CachingService, create_caching_service)

        logger.info("Dependency registry setup completed successfully")
    except Exception as e:
        logger.exception(
            f"Failed to setup dependency registry: {str(e)}", exc_info=True
        )
        raise


# Convenience functions for service resolution
def get_llm_service() -> LLMService:
    """Get or create an LLM service instance."""
    return dependency_registry.resolve(LLMService)


def get_service_manager() -> BaseServiceManager:
    """Get or create a service manager instance."""
    return dependency_registry.resolve(BaseServiceManager)


def get_config_manager() -> ConfigurationManager:
    """Get the configuration manager instance."""
    return dependency_registry.resolve(ConfigurationManager)


def get_vdb_service() -> VectorDBService:
    """Get or create a VectorDB service instance."""
    return dependency_registry.resolve(VectorDBService)


def get_embedding_service() -> EmbeddingService:
    """Get or create an embedding service instance."""
    return dependency_registry.resolve(EmbeddingService)


def get_search_service() -> SearchService:
    """Get or create a Search service instance."""
    return dependency_registry.resolve(SearchService)


def get_caching_service() -> CachingService:
    """Get or create a caching service instance."""
    return dependency_registry.resolve(CachingService)
