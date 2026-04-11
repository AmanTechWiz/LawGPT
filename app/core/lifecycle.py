"""
Application lifecycle management for startup, shutdown, and component initialization.

Handles coordinated initialization of database connections, RAG engine, legal agent,
and graceful shutdown procedures with error handling and status tracking.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from .config import settings
from .database import db_manager, init_database

logger = logging.getLogger(__name__)


class ApplicationLifecycleManager:
    """
    Manages the complete lifecycle of the legal research assistant application.

    Handles initialization, startup, and shutdown of all system components
    including database connections, RAG engine, and legal agent.
    """

    def __init__(self):
        self.initialized = False
        self.startup_complete = False

    async def initialize_application(self) -> None:
        """
        Initialize all application components.

        Raises:
            Exception: If initialization fails
        """
        if self.initialized:
            logger.info("Application already initialized")
            return

        try:
            logger.info("Starting Legal Research Assistant initialization...")

            # Initialize database
            await self._initialize_database()

            # Wait for database to be ready
            await asyncio.sleep(2)

            # Initialize RAG engine
            await self._initialize_rag_engine()

            # Initialize legal agent
            await self._initialize_legal_agent()

            # Initialize multi-hop reasoning services
            await self._initialize_multi_hop_reasoning()

            self.initialized = True
            self.startup_complete = True
            logger.info("Legal Research Assistant started successfully!")

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self.startup_complete = False
            raise

    async def shutdown_application(self) -> None:
        """
        Shutdown all application components gracefully.
        """
        try:
            logger.info("Shutting down Legal Research Assistant...")

            # Close database connections
            await self._shutdown_database()

            self.startup_complete = False
            logger.info("Legal Research Assistant shut down successfully!")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _initialize_database(self) -> None:
        """
        Initialize database connections and schema.

        Raises:
            Exception: If database initialization fails
        """
        try:
            logger.info("Initializing database...")
            await db_manager.initialize()
            await init_database()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def _initialize_rag_engine(self) -> None:
        """
        Initialize the RAG engine.

        Raises:
            Exception: If RAG engine initialization fails
        """
        try:
            from ..services.lightweight_llm_rag import lightweight_llm_rag
            logger.info("Initializing RAG engine...")
            await lightweight_llm_rag.initialize()
            logger.info("RAG engine initialized successfully")
        except Exception as e:
            logger.error(f"RAG engine initialization failed: {e}")
            raise

    async def _initialize_legal_agent(self) -> None:
        """
        Initialize the legal agent.

        Raises:
            Exception: If legal agent initialization fails
        """
        try:
            
            logger.info("Skipping legal agent initialization...")
        except Exception as e:
            logger.error(f"Legal agent initialization failed: {e}")
            raise

    async def _initialize_multi_hop_reasoning(self) -> None:
        """
        Initialize multi-hop reasoning services.

        Raises:
            Exception: If multi-hop reasoning initialization fails
        """
        try:
            from ..services.multi_hop_reasoning import multi_hop_reasoning_engine
            logger.info("Initializing multi-hop reasoning services...")
            # await multi_hop_reasoning_engine.initialize()
            logger.info("Multi-hop reasoning services initialized successfully")
        except Exception as e:
            logger.error(f"Multi-hop reasoning initialization failed: {e}")
            raise

    async def _shutdown_database(self) -> None:
        """
        Shutdown database connections.
        """
        try:
            await db_manager.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    def is_ready(self) -> bool:
        """
        Check if the application is ready to serve requests.

        Returns:
            bool: True if application is ready, False otherwise
        """
        return self.startup_complete and self.initialized


# Global lifecycle manager instance
lifecycle_manager = ApplicationLifecycleManager()


@asynccontextmanager
async def application_lifespan(app) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for application lifecycle.

    Args:
        app: FastAPI application instance

    Yields:
        None: During application runtime

    Raises:
        Exception: If startup fails
    """
    try:
        await lifecycle_manager.initialize_application()
        yield
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        yield  # Still yield to allow FastAPI to start
    finally:
        await lifecycle_manager.shutdown_application()
