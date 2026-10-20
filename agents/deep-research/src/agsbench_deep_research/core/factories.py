"""
Factory functions for creating service instances.

This module centralizes all service creation logic, providing
factory functions that create properly configured service instances
based on application configuration.
"""

from .service.llm import (
    LLMService,
    LLMServiceInstanceConfig,
    LLMServiceRemoteInstance,
    LLMServiceAnthropicInstance,
)
from .service.embedding import (
    EmbeddingService,
    EmbeddingServiceInstanceConfig,
    EmbeddingServiceRemoteInstance,
    EmbeddingServiceLocalInstance,
)
from .service.vdb import (
    VectorDBService,
    VectorDBServiceInstanceConfig,
    MilvusServiceInstance,
    MilvusLiteServiceInstance,
    SimpleVectorDBServiceInstance,
)
from .service.search import (
    SearchService,
    SearchServiceInstanceConfig,
    TavilySearchServiceInstance,
    ArxivSearchServiceInstance,
    DuckDuckGoSearchServiceInstance,
    WikiSearchServiceInstance,
    PerplexitySearchServiceInstance,
    GoogleSearchServiceInstance,
    DummySearchServiceInstance,
    ExaSearchServiceInstance,
    LinkupSearchServiceInstance,
    SerperSearchServiceInstance,
    SearxNGSearchServiceInstance,
    FirecrawlSearchServiceInstance,
)
from .service.caching import (
    CachingService,
    CachingServiceInstanceConfig,
    InMemoryCacheServiceInstance,
    FileSystemCacheServiceInstance,
    get_cache_instance_class,
)
from typing import Dict, Any, List, Optional
from .services_manager import BaseServiceManager
from .config import config_manager
import logging

factories_logger = logging.getLogger(__name__)


def create_llm_service() -> LLMService:
    """Create a new LLM service with configuration from config files.

    This factory function creates an LLM service with instances configured
    according to the core.service.llm configuration file, handling different
    instance types based on name prefixes.

    Returns:
        A fully configured LLM service with all enabled instances
    """

    # Get LLM configuration
    llm_config = config_manager.get_config("core.service.llm")
    service = LLMService(
        scheduling_policy=llm_config.get("scheduling_policy", "round_robin")
    )

    # Create instances from configuration
    instances: List[Dict[str, Any]] = llm_config.get("instances", [])
    for instance_config in instances:
        # Skip disabled instances
        if not instance_config.get("enabled", True):
            continue

        name: str = instance_config.get("name")
        if not name:
            continue

        group_name: str = instance_config.get("group_name", "default")

        # Create tags set including the model name for backward compatibility
        tags = set(instance_config.get("tags", []))
        tags.add(
            instance_config.get("model", llm_config.get("default_model"))
        )  # Add model as a tag for routing

        config = LLMServiceInstanceConfig(
            location=instance_config.get("location", "local:auto"),
            api_key=instance_config.get("api_key", None),
            model=instance_config.get("model", llm_config.get("default_model")),
            tags=frozenset(tags),
        )

        # Determine instance type based on name prefix
        try:
            if name.startswith("anthropic:"):
                instance = LLMServiceAnthropicInstance(name, group_name, config)
            elif name.startswith("vllm:"):
                instance = LLMServiceRemoteInstance(name, group_name, config)
            else:
                instance = LLMServiceRemoteInstance(name, group_name, config)
        except Exception as e:
            factories_logger.exception(
                f"Failed to create LLM instance {name}: {str(e)}", exc_info=True
            )
            continue

        # For remote API endpoints (e.g. DashScope), the /health endpoint
        # may not exist.  When the location contains "://" we skip the
        # health gate and always add the instance.
        skip_health = "://" in instance_config.get("location", "")
        if skip_health or instance.is_healthy():
            factories_logger.info(
                f"Initialized LLM instance: {name}, adding to service"
                + (" (health check skipped for remote API)" if skip_health else "")
            )
            service.add_instance(
                instance_name=instance.instance_name, instance=instance
            )
        else:
            factories_logger.warning(
                f"LLM instance {name} is not healthy and will not be added to the service"
            )

    return service


def create_vdb_service() -> VectorDBService:
    """Create a new VectorDB service with configuration from config files.

    This factory function creates a VectorDB service with instances configured
    according to the core.service.vdb configuration file, handling different
    instance types based on db_type.

    Returns:
        A fully configured VectorDB service with all enabled instances
    """

    try:
        # Get VDB configuration
        vdb_config = config_manager.get_config("core.service.vdb")
        scheduling_policy = vdb_config.get("scheduling_policy", "round_robin")
        collection_assignment_strategy = vdb_config.get(
            "collection_assignment_strategy", "consistent_hashing"
        )
        service = VectorDBService(
            scheduling_policy=scheduling_policy,
            collection_assignment_strategy=collection_assignment_strategy,
        )

        # Create instances from configuration
        instances: List[Dict[str, Any]] = vdb_config.get("instances", [])
        for instance_config in instances:
            # Skip disabled instances
            if not instance_config.get("enabled", True):
                continue

            name: str = instance_config.get("name")
            if not name:
                continue

            group_name: str = instance_config.get("group_name", "default")

            # Create tags set including the model name for backward compatibility
            tags = set(instance_config.get("tags", []))
            # VDB instances don't have models, so we don't add model as a tag

            # Create instance config with default values
            config = VectorDBServiceInstanceConfig(
                db_type=instance_config.get("db_type"),
                location=instance_config.get("location", "local"),
                collection_name=instance_config.get(
                    "collection_name",
                    vdb_config.get("default_collection", "default_collection"),
                ),
                dim=int(
                    instance_config.get("dim", vdb_config.get("default_dim", 1024))
                ),
                metric_type=instance_config.get(
                    "metric_type", vdb_config.get("metric_type", "L2")
                ),
                index_params=instance_config.get("index_params", {}),
                search_params=instance_config.get("search_params", {}),
                tags=frozenset(tags),
            )

            # Determine instance type based on db_type
            try:
                db_type = instance_config.get("db_type")
                if db_type == "milvus":
                    instance = MilvusServiceInstance(name, group_name, config)
                elif db_type == "milvus_lite":
                    instance = MilvusLiteServiceInstance(name, group_name, config)
                elif db_type == "simple":
                    instance = SimpleVectorDBServiceInstance(name, group_name, config)
                else:
                    factories_logger.warning(f"Unsupported VectorDB type: {db_type}")
                    continue
            except Exception as e:
                factories_logger.exception(
                    f"Failed to create VectorDB instance {name}: {str(e)}",
                    exc_info=True,
                )
                continue

            service.add_instance(
                instance_name=instance.instance_name, instance=instance
            )
            factories_logger.info(f"Initialized VectorDB instance: {name} ({db_type})")

    except Exception as e:
        factories_logger.exception(
            f"Failed to initialize VectorDB service: {str(e)}", exc_info=True
        )

    return service


def create_embedding_service() -> EmbeddingService:
    """Create a new Embedding service with configuration from config files.

    This factory function creates an Embedding service with instances configured
    according to the core.service.embedding configuration file, handling different
    instance types based on name prefixes.

    Returns:
        A fully configured Embedding service with all enabled instances
    """

    # Get Embedding configuration
    embedding_config = config_manager.get_config("core.service.embedding")
    service = EmbeddingService(
        scheduling_policy=embedding_config.get("scheduling_policy", "round_robin")
    )

    # Create instances from configuration
    instances: List[Dict[str, Any]] = embedding_config.get("instances", [])
    for instance_config in instances:
        # Skip disabled instances
        if not instance_config.get("enabled", True):
            continue

        name: str = instance_config.get("name")
        if not name:
            continue

        group_name: str = instance_config.get("group_name", "default")

        # Create tags set including the model name for backward compatibility
        tags = set(instance_config.get("tags", []))
        tags.add(
            instance_config.get("model", embedding_config.get("default_model"))
        )  # Add model as a tag for routing

        config = EmbeddingServiceInstanceConfig(
            location=instance_config.get("location", "local:auto"),
            api_key=instance_config.get("api_key", None),
            model=instance_config.get("model", embedding_config.get("default_model")),
            tags=frozenset(tags),
        )

        # Determine instance type based on name prefix
        try:
            if name.startswith("vllm:"):
                instance = EmbeddingServiceRemoteInstance(name, group_name, config)
            elif name.startswith("local"):
                instance = EmbeddingServiceLocalInstance(name, group_name, config)
            else:
                instance = EmbeddingServiceRemoteInstance(name, group_name, config)
        except Exception as e:
            factories_logger.exception(
                f"Failed to create Embedding instance {name}: {str(e)}", exc_info=True
            )
            continue

        service.add_instance(instance_name=instance.instance_name, instance=instance)

    return service


def create_search_service() -> SearchService:
    """Create a new Search service with configuration from config files.

    This factory function creates a Search service with instances configured
    according to the core.service.search configuration file, handling different
    instance types based on engine name.

    Returns:
        A fully configured Search service with all enabled instances
    """
    search_config = config_manager.get_config("core.service.search")
    service = SearchService(
        scheduling_policy=search_config.get("scheduling_policy", "round_robin")
    )
    instances: List[Dict[str, Any]] = search_config.get("instances", [])
    for instance_config in instances:
        # Skip disabled instances
        if not instance_config.get("enabled", True):
            continue

        name: str = instance_config.get("name")
        if not name:
            continue

        group_name: str = instance_config.get("group_name", "default")

        # Create tags set including the model name for backward compatibility
        tags = set(instance_config.get("tags", []))
        tags.add(
            instance_config.get("engine", search_config.get("default_engine", "tavily"))
        )

        instance_caching_config = instance_config.get("caching", {})

        config = SearchServiceInstanceConfig(
            location=instance_config.get("location", "local:auto"),
            api_key=instance_config.get("api_key", None),
            engine=instance_config.get(
                "engine", search_config.get("default_engine", "tavily")
            ),
            caching_config=instance_caching_config,
            tags=frozenset(tags),
            kwargs=instance_config.get("kwargs", {}),
        )

        try:
            engine = config.engine
            if engine == "tavily":
                instance = TavilySearchServiceInstance(name, group_name, config)
            elif engine == "arxiv":
                instance = ArxivSearchServiceInstance(name, group_name, config)
            elif engine == "duckduckgo":
                instance = DuckDuckGoSearchServiceInstance(name, group_name, config)
            elif engine == "wiki":
                instance = WikiSearchServiceInstance(name, group_name, config)
            elif engine == "perplexity":
                instance = PerplexitySearchServiceInstance(name, group_name, config)
            elif engine == "googlesearch":
                instance = GoogleSearchServiceInstance(name, group_name, config)
            elif engine == "dummy":
                instance = DummySearchServiceInstance(name, group_name, config)
            elif engine == "exa":
                instance = ExaSearchServiceInstance(name, group_name, config)
            elif engine == "linkup":
                instance = LinkupSearchServiceInstance(name, group_name, config)
            elif engine == "serper":
                instance = SerperSearchServiceInstance(name, group_name, config)
            elif engine == "searxng":
                instance = SearxNGSearchServiceInstance(name, group_name, config)
            elif engine == "firecrawl":
                instance = FirecrawlSearchServiceInstance(name, group_name, config)
            # Add more search engines as needed
            else:
                factories_logger.warning(f"Unsupported search engine: {engine}")
                continue
        except Exception as e:
            factories_logger.exception(
                f"Failed to create Search instance {name}: {str(e)}", exc_info=True
            )
            continue

        service.add_instance(instance_name=instance.instance_name, instance=instance)

    return service


def create_caching_service() -> CachingService:
    """Create a new Caching service with configuration from config files.

    This factory function creates a Caching service with instances configured
    according to the core.service.caching configuration file, handling different
    instance types based on backend type.

    Returns:
        A fully configured Caching service with all enabled instances
    """
    caching_config = config_manager.get_config("core.service.caching")
    service = CachingService(
        scheduling_policy=caching_config.get("scheduling_policy", "round_robin"),
        stateful_assignment_strategy=caching_config.get(
            "stateful_assignment_strategy", "load_balanced"
        ),
    )
    instances: List[Dict[str, Any]] = caching_config.get("instances", [])
    for instance_config in instances:
        if not instance_config.get("enabled", True):
            continue
        name: str = instance_config.get("name")
        if not name:
            continue
        group_name: str = instance_config.get("group_name", "default")
        backend = instance_config.get("backend", "memory")
        tags = set(instance_config.get("tags", []))
        tags.add(backend)  # Add backend as a tag for routing
        config = CachingServiceInstanceConfig(
            backend=backend,
            location=instance_config.get("location", None),
            default_ttl=instance_config.get("default_ttl", None),
            tags=frozenset(tags),
        )
        try:
            instance_class = get_cache_instance_class(backend)
            instance = instance_class(name, group_name, config)
        except Exception as e:
            factories_logger.exception(
                f"Failed to create Caching instance {name}: {str(e)}", exc_info=True
            )
            continue
        service.add_instance(instance_name=instance.instance_name, instance=instance)
    return service


def create_service_manager() -> BaseServiceManager:
    """Create a new service manager with default services."""
    from .di import dependency_registry

    manager_cfg = config_manager.get_config("core.service_manager", {})
    # Initialize the service manager
    manager = BaseServiceManager(
        health_check_interval=manager_cfg.get("health_check_interval", 30),
        max_connections=manager_cfg.get("max_connections", 100),
    )

    # Add LLM service
    try:
        llm_service = dependency_registry.resolve(LLMService)
        manager.add_service(llm_service.service_name, llm_service)
    except Exception as e:
        factories_logger.exception(
            f"Failed to initialize LLM service: {str(e)}", exc_info=True
        )

    # Add VectorDB service
    try:
        vdb_service = dependency_registry.resolve(VectorDBService)
        manager.add_service(vdb_service.service_name, vdb_service)
    except Exception as e:
        factories_logger.exception(
            f"Failed to initialize VectorDB service: {str(e)}", exc_info=True
        )

    # Add Embedding service
    try:
        embedding_service = dependency_registry.resolve(EmbeddingService)
        manager.add_service(embedding_service.service_name, embedding_service)
    except Exception as e:
        factories_logger.exception(
            f"Failed to initialize Embedding service: {str(e)}", exc_info=True
        )

    # Add Search service
    try:
        search_service = dependency_registry.resolve(SearchService)
        manager.add_service(search_service.service_name, search_service)
    except Exception as e:
        factories_logger.exception(
            f"Failed to initialize Search service: {str(e)}", exc_info=True
        )

    # Add Caching service
    try:
        caching_service = dependency_registry.resolve(CachingService)
        manager.add_service(caching_service.service_name, caching_service)
    except Exception as e:
        factories_logger.exception(
            f"Failed to initialize Caching service: {str(e)}", exc_info=True
        )

    return manager
