"""
Configuration management using Pydantic Settings.

Supports loading from environment variables and YAML files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "liteads"
    user: str = "liteads"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def async_url(self) -> str:
        """Get async database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """Get sync database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    pool_size: int = 10

    @property
    def url(self) -> str:
        """Get Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False


class AdServingSettings(BaseSettings):
    """Ad serving configuration."""

    default_num_ads: int = 1
    max_num_ads: int = 10
    timeout_ms: int = 50
    enable_ml_prediction: bool = True


class FrequencySettings(BaseSettings):
    """Frequency control configuration."""

    default_daily_cap: int = 3
    default_hourly_cap: int = 1
    ttl_hours: int = 24


class MLSettings(BaseSettings):
    """ML model configuration."""

    model_dir: str = "./models"
    ctr_model: str = "deepfm_v1"
    cvr_model: str = "deepfm_cvr_v1"
    embedding_dim: int = 8
    batch_size: int = 128


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    format: Literal["json", "console"] = "json"


class MonitoringSettings(BaseSettings):
    """Monitoring configuration."""

    enabled: bool = True
    prometheus_port: int = 9090


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="LITEADS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application
    app_name: str = "LiteAds"
    app_version: str = "0.1.0"
    debug: bool = False
    env: Literal["dev", "prod", "test"] = "dev"

    # Nested settings
    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ad_serving: AdServingSettings = Field(default_factory=AdServingSettings)
    frequency: FrequencySettings = Field(default_factory=FrequencySettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ("dev", "prod", "test"):
            raise ValueError(f"Invalid environment: {v}")
        return v


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings.

    Loads from:
    1. configs/base.yaml (base configuration)
    2. configs/{env}.yaml (environment-specific overrides)
    3. Environment variables (highest priority)
    """
    import os

    env = os.getenv("LITEADS_ENV", "dev")

    # Find config directory
    config_dir = Path(__file__).parent.parent.parent / "configs"

    # Load base config
    base_config = load_yaml_config(config_dir / "base.yaml")

    # Load environment-specific config
    env_config = load_yaml_config(config_dir / f"{env}.yaml")

    # Merge configurations
    merged = merge_configs(base_config, env_config)

    # Flatten nested config for Pydantic
    flat_config = {}
    if "app" in merged:
        flat_config["app_name"] = merged["app"].get("name", "LiteAds")
        flat_config["app_version"] = merged["app"].get("version", "0.1.0")
        flat_config["debug"] = merged["app"].get("debug", False)

    flat_config["env"] = env

    # Create nested settings
    if "server" in merged:
        flat_config["server"] = ServerSettings(**merged["server"])
    if "database" in merged:
        flat_config["database"] = DatabaseSettings(**merged["database"])
    if "redis" in merged:
        flat_config["redis"] = RedisSettings(**merged["redis"])
    if "ad_serving" in merged:
        flat_config["ad_serving"] = AdServingSettings(**merged["ad_serving"])
    if "frequency" in merged:
        flat_config["frequency"] = FrequencySettings(**merged["frequency"])
    if "ml" in merged:
        flat_config["ml"] = MLSettings(**merged["ml"])
    if "logging" in merged:
        flat_config["logging"] = LoggingSettings(**merged["logging"])
    if "monitoring" in merged:
        flat_config["monitoring"] = MonitoringSettings(**merged["monitoring"])

    return Settings(**flat_config)


# Convenience alias
settings = get_settings()
