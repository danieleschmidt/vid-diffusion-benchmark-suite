"""Database connection management."""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL. If None, reads from environment.
        """
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
        self._initialize_engine()
        
    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try PostgreSQL first
        if all(key in os.environ for key in ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER']):
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB', 'vid_bench')
            user = os.getenv('POSTGRES_USER', 'vid_bench')
            password = os.getenv('POSTGRES_PASSWORD', '')
            
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
            
        # Fallback to SQLite
        db_path = os.getenv('DATABASE_PATH', './vid_bench.db')
        return f"sqlite:///{db_path}"
        
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine."""
        engine_kwargs = {
            'echo': os.getenv('VID_BENCH_LOG_LEVEL') == 'DEBUG',
            'future': True
        }
        
        # Configure connection pool for PostgreSQL
        if self.database_url.startswith('postgresql'):
            engine_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
                'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '10')),
                'pool_pre_ping': True,
                'pool_recycle': 3600  # 1 hour
            })
            
        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.session_factory = sessionmaker(bind=self.engine)
        
        logger.info(f"Database engine initialized: {self.database_url.split('://')[0]}")
        
    def create_all_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")
        
    def drop_all_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped successfully")
        
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around database operations."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.session_factory()
        
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.session_scope() as session:
                result = session.execute(text("SELECT 1")).scalar()
                
                # Get database statistics
                stats = {
                    "status": "healthy",
                    "database_type": self.database_url.split('://')[0],
                    "connection_test": result == 1
                }
                
                # Add PostgreSQL-specific stats
                if self.database_url.startswith('postgresql'):
                    try:
                        # Get connection count
                        conn_result = session.execute(
                            text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                        ).scalar()
                        stats["active_connections"] = conn_result
                        
                        # Get database size
                        size_result = session.execute(
                            text("SELECT pg_size_pretty(pg_database_size(current_database()))")
                        ).scalar()
                        stats["database_size"] = size_result
                        
                    except Exception as e:
                        logger.debug(f"Failed to get PostgreSQL stats: {e}")
                        
                return stats
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_type": self.database_url.split('://')[0]
            }
            
    def execute_migration(self, migration_sql: str):
        """Execute a database migration."""
        try:
            with self.session_scope() as session:
                session.execute(text(migration_sql))
            logger.info("Migration executed successfully")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
            
    def backup_database(self, backup_path: str):
        """Create database backup (PostgreSQL only)."""
        if not self.database_url.startswith('postgresql'):
            raise NotImplementedError("Backup only supported for PostgreSQL")
            
        # Extract connection details
        import urllib.parse
        parsed = urllib.parse.urlparse(self.database_url)
        
        cmd = [
            'pg_dump',
            '-h', parsed.hostname,
            '-p', str(parsed.port or 5432),
            '-U', parsed.username,
            '-d', parsed.path[1:],  # Remove leading /
            '-f', backup_path,
            '--no-password'
        ]
        
        env = os.environ.copy()
        env['PGPASSWORD'] = parsed.password
        
        import subprocess
        try:
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"Database backup created: {backup_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Database backup failed: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session() -> Session:
    """Get database session (convenience function)."""
    return db_manager.get_session()


@contextmanager
def db_session_scope():
    """Get transactional database session scope (convenience function)."""
    with db_manager.session_scope() as session:
        yield session