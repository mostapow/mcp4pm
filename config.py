import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    host: str = Field(default=os.getenv('DB_HOST', 'localhost'), description="Database host")
    port: int = Field(default=int(os.getenv('DB_PORT', 5432)), description="Database port")
    database: str = Field(default=os.getenv('DB_NAME', 'mcp'), description="Database name")
    username: str = Field(default=os.getenv('DB_USER', 'postgres'), description="Username")
    password: str = Field(default=os.getenv('DB_PASSWORD', ''), description="Password")
    db_schema: Optional[str] = Field(default=os.getenv('DB_SCHEMA', 'public'), description="Database schema")

    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class TableSchema(BaseModel):
    """Schema definition for event log table"""
    table_name: str = Field(default=os.getenv('TABLE_NAME', 'bpi2017'), description="Name of the event log table")
    case_column: str = Field(default=os.getenv('CASE_COLUMN', '\"case:concept:name\"'), description="Column containing case IDs")
    activity_column: str = Field(default=os.getenv('ACTIVITY_COLUMN', '\"concept:name\"'), description="Column containing activity names")
    timestamp_column: str = Field(default=os.getenv('TIMESTAMP_COLUMN', '\"time:timestamp\"'), description="Column containing timestamps")
    resource_column: Optional[str] = Field(default=os.getenv('RESOURCE_COLUMN', None), description="Column containing resources")

    # Additional columns to include
    additional_columns: Optional[List[str]] = Field(
        default=None,
        description="Additional columns to include in the event log"
    )

    # Optional filters
    where_clause: Optional[str] = Field(
        default=None,
        description="SQL WHERE clause to filter events (without WHERE keyword)"
    )

    sample_percentage: Optional[float] = Field(
        default=100.00,
        ge=0.0,
        le=100.0,
        description="Random sample percentage of cases"
    )


class ServerConfig(BaseModel):
    """Server configuration"""
    anthropic_api_key: str = Field(default=os.getenv('ANTHROPIC_API_KEY', ''), description="Anthropic API key")
    server_name: str = Field(default=os.getenv('SERVER_NAME', 'bpi2017'), description="Server name")
    top_k_results: int = Field(default=int(os.getenv('TOP_K_RESULTS', 10)), description="Default number of results to return")


# Create configuration instances
DB_CONFIG = DatabaseConfig()
TABLE_SCHEMA = TableSchema()
SERVER_CONFIG = ServerConfig()

# Create database URL for SQLAlchemy
DATABASE_URL = DB_CONFIG.connection_string