"""
Main Configuration File
Centralized configuration for the institutional trading system
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_db"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class APIConfig:
    """API configuration"""
    # Exchange APIs
    binance_api_key: str = ""
    binance_api_secret: str = ""
    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    kraken_api_key: str = ""
    kraken_api_secret: str = ""
    
    # Data APIs
    alphavantage_api_key: str = ""
    polygon_api_key: str = ""
    tiingo_api_key: str = ""
    
    # Rate limits
    requests_per_minute: int = 60
    requests_per_second: int = 10
    
    # Timeouts
    connection_timeout: int = 30
    read_timeout: int = 30


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Position limits
    max_position_size: float = 1000000.0  # USD
    max_symbol_exposure: float = 0.2  # 20% of portfolio
    max_sector_exposure: float = 0.4  # 40% of portfolio
    
    # Risk limits
    max_daily_loss: float = 0.05  # 5% of portfolio
    max_drawdown: float = 0.15  # 15% of portfolio
    var_95_limit: float = 0.03  # 3% of portfolio
    
    # Margin requirements
    initial_margin: float = 0.25  # 25% initial margin
    maintenance_margin: float = 0.15  # 15% maintenance margin
    
    # Volatility adjustments
    volatility_scaling: bool = True
    volatility_lookback: int = 20  # days
    volatility_threshold: float = 0.5  # 50% above normal
    
    # Circuit breakers
    enable_circuit_breakers: bool = True
    loss_circuit_breaker: float = 0.1  # 10% loss triggers stop
    time_circuit_breaker: int = 3600  # 1 hour max position time


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    # General
    enabled_strategies: list = field(default_factory=lambda: [
        'smart_money',
        'liquidity_grab',
        'market_profile'
    ])
    
    # Smart Money
    smart_money_enabled: bool = True
    smart_money_allocation: float = 0.3  # 30% of capital
    wyckoff_lookback: int = 100
    vsa_volume_threshold: float = 2.0
    
    # Liquidity Grab
    liquidity_grab_enabled: bool = True
    liquidity_grab_allocation: float = 0.3
    liquidity_threshold: float = 3.0
    stop_hunt_sensitivity: float = 2.0
    
    # Market Profile
    market_profile_enabled: bool = True
    market_profile_allocation: float = 0.2
    poc_sensitivity: float = 0.01
    value_area_threshold: float = 0.7
    
    # Common parameters
    max_positions_per_strategy: int = 5
    position_sizing_method: str = 'kelly'  # 'fixed', 'percent', 'kelly'
    default_position_size: float = 0.1  # 10% of capital


@dataclass
class ExecutionConfig:
    """Execution configuration"""
    # Order types
    use_iceberg_orders: bool = True
    use_twap_orders: bool = True
    use_vwap_orders: bool = True
    
    # Slippage control
    max_slippage_bps: int = 10  # 10 basis points
    use_smart_routing: bool = True
    
    # Transaction cost analysis
    enable_tca: bool = True
    tca_lookback: int = 100
    
    # Order management
    order_expiry_seconds: int = 300  # 5 minutes
    max_order_retries: int = 3
    retry_delay_seconds: int = 1
    
    # Multi-broker execution
    primary_broker: str = 'binance'
    backup_brokers: list = field(default_factory=lambda: ['coinbase', 'kraken'])
    broker_failover_threshold: int = 3  # failures before failover


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    # Dashboard
    dashboard_enabled: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8050
    dashboard_update_interval: float = 1.0  # seconds
    
    # Alerts
    alert_channels: list = field(default_factory=lambda: ['console', 'log'])
    alert_thresholds: dict = field(default_factory=lambda: {
        'position_concentration': 0.7,
        'margin_usage': 0.5,
        'daily_loss': 0.03,
        'var_breach': 0.05
    })
    
    # Performance monitoring
    performance_update_interval: int = 3600  # 1 hour
    attribution_period: str = 'daily'  # 'hourly', 'daily', 'weekly'
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_retention_days: int = 30


@dataclass
class CacheConfig:
    """Cache configuration"""
    # Data cache
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    max_cache_size_mb: int = 1024  # 1 GB
    default_ttl_minutes: int = 60
    
    # Redis cache (if available)
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Memory cache
    memory_cache_size: int = 1000  # items


@dataclass
class BacktestingConfig:
    """Backtesting configuration"""
    # Data
    data_source: str = "binance"
    data_dir: str = "data/historical"
    data_format: str = "parquet"  # 'csv', 'parquet', 'feather'
    
    # Simulation
    initial_capital: float = 1000000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "fixed"  # 'fixed', 'proportional', 'random'
    slippage_bps: int = 5
    
    # Walk-forward optimization
    wfo_enabled: bool = True
    wfo_window_size: int = 100  # days
    wfo_step_size: int = 20  # days
    wfo_optimization_period: int = 10  # days
    
    # Metrics
    calculate_advanced_metrics: bool = True
    risk_free_rate: float = 0.02  # 2% annual


@dataclass
class SystemConfig:
    """Main system configuration"""
    # System identification
    system_name: str = "InstitutionalTradingSystem"
    version: str = "1.0.0"
    environment: str = "development"  # 'development', 'staging', 'production'
    
    # Core components
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    
    # Trading parameters
    symbols: list = field(default_factory=lambda: [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'SOL/USDT',
        'XRP/USDT'
    ])
    
    trading_hours: dict = field(default_factory=lambda: {
        'start': '00:00',
        'end': '23:59',
        'timezone': 'UTC'
    })
    
    # System behavior
    enable_live_trading: bool = False
    paper_trading: bool = True
    heartbeat_interval: int = 60  # seconds
    max_workers: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: str):
        """Save configuration to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}, using defaults")
            return cls()
        
        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif filepath.suffix in ['.yaml', '.yml']:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            # Convert nested dictionaries to dataclasses
            config = cls()
            
            # Update database config
            if 'database' in data:
                for key, value in data['database'].items():
                    setattr(config.database, key, value)
            
            # Update API config
            if 'api' in data:
                for key, value in data['api'].items():
                    setattr(config.api, key, value)
            
            # Update risk config
            if 'risk' in data:
                for key, value in data['risk'].items():
                    setattr(config.risk, key, value)
            
            # Update strategy config
            if 'strategy' in data:
                for key, value in data['strategy'].items():
                    setattr(config.strategy, key, value)
            
            # Update execution config
            if 'execution' in data:
                for key, value in data['execution'].items():
                    setattr(config.execution, key, value)
            
            # Update monitoring config
            if 'monitoring' in data:
                for key, value in data['monitoring'].items():
                    setattr(config.monitoring, key, value)
            
            # Update cache config
            if 'cache' in data:
                for key, value in data['cache'].items():
                    setattr(config.cache, key, value)
            
            # Update backtesting config
            if 'backtesting' in data:
                for key, value in data['backtesting'].items():
                    setattr(config.backtesting, key, value)
            
            # Update top-level attributes
            for key, value in data.items():
                if key not in ['database', 'api', 'risk', 'strategy', 
                              'execution', 'monitoring', 'cache', 'backtesting']:
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            logger.info(f"Configuration loaded from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return cls()


def load_environment_config() -> SystemConfig:
    """Load configuration from environment variables"""
    config = SystemConfig()
    
    # Override from environment variables
    # Database
    config.database.host = os.getenv('DB_HOST', config.database.host)
    config.database.port = int(os.getenv('DB_PORT', config.database.port))
    config.database.database = os.getenv('DB_NAME', config.database.database)
    config.database.username = os.getenv('DB_USER', config.database.username)
    config.database.password = os.getenv('DB_PASS', config.database.password)
    
    # API keys
    config.api.binance_api_key = os.getenv('BINANCE_API_KEY', config.api.binance_api_key)
    config.api.binance_api_secret = os.getenv('BINANCE_API_SECRET', config.api.binance_api_secret)
    config.api.coinbase_api_key = os.getenv('COINBASE_API_KEY', config.api.coinbase_api_key)
    config.api.coinbase_api_secret = os.getenv('COINBASE_API_SECRET', config.api.coinbase_api_secret)
    config.api.kraken_api_key = os.getenv('KRAKEN_API_KEY', config.api.kraken_api_key)
    config.api.kraken_api_secret = os.getenv('KRAKEN_API_SECRET', config.api.kraken_api_secret)
    config.api.alphavantage_api_key = os.getenv('ALPHAVANTAGE_API_KEY', config.api.alphavantage_api_key)
    
    # System
    config.environment = os.getenv('ENVIRONMENT', config.environment)
    config.enable_live_trading = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
    config.paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    # Monitoring
    config.monitoring.log_level = os.getenv('LOG_LEVEL', config.monitoring.log_level)
    config.monitoring.dashboard_host = os.getenv('DASHBOARD_HOST', config.monitoring.dashboard_host)
    config.monitoring.dashboard_port = int(os.getenv('DASHBOARD_PORT', config.monitoring.dashboard_port))
    
    return config


def get_config(config_file: Optional[str] = None) -> SystemConfig:
    """Get configuration, loading from file if provided"""
    if config_file:
        config = SystemConfig.load(config_file)
    else:
        # Try to load from default locations
        default_paths = [
            'config/config.json',
            'config/config.yaml',
            'config.json',
            'config.yaml'
        ]
        
        for path in default_paths:
            if Path(path).exists():
                config = SystemConfig.load(path)
                break
        else:
            # Load from environment
            config = load_environment_config()
    
    # Apply environment-specific overrides
    if config.environment == 'production':
        # Production-specific settings
        config.monitoring.log_level = 'WARNING'
        config.enable_live_trading = True
        config.paper_trading = False
        config.risk.max_daily_loss = 0.02  # More conservative in production
        config.risk.max_drawdown = 0.10
    
    elif config.environment == 'staging':
        # Staging-specific settings
        config.monitoring.log_level = 'INFO'
        config.enable_live_trading = False
        config.paper_trading = True
        config.risk.max_daily_loss = 0.03
    
    # Development uses default settings
    
    return config


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_global_config() -> SystemConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def set_global_config(config: SystemConfig):
    """Set global configuration instance"""
    global _config
    _config = config
    logger.info("Global configuration updated")


# Convenience functions
def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_global_config().database


def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_global_config().api


def get_risk_config() -> RiskConfig:
    """Get risk configuration"""
    return get_global_config().risk


def get_strategy_config() -> StrategyConfig:
    """Get strategy configuration"""
    return get_global_config().strategy


def get_execution_config() -> ExecutionConfig:
    """Get execution configuration"""
    return get_global_config().execution


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_global_config().monitoring


def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return get_global_config().cache


def get_backtesting_config() -> BacktestingConfig:
    """Get backtesting configuration"""
    return get_global_config().backtesting


# Example usage
if __name__ == '__main__':
    # Create default configuration
    config = SystemConfig()
    
    # Save to file
    config.save('config/config.json')
    config.save('config/config.yaml')
    
    # Load configuration
    loaded_config = SystemConfig.load('config/config.json')
    
    # Print configuration
    print("Configuration:")
    print(loaded_config.to_json())
    
    # Get specific config sections
    print("\nDatabase URL:", loaded_config.database.url)
    print("Enabled strategies:", loaded_config.strategy.enabled_strategies)