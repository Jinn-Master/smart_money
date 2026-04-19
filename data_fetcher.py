"""
Intelligent Data Fetcher
Handles market data fetching with caching, retries, and rate limiting
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import gzip
import os

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    """Data request specification"""
    source: str  # 'binance', 'coinbase', 'kraken', 'bybit', 'alphavantage'
    symbol: str
    data_type: str  # 'ohlcv', 'trades', 'orderbook', 'funding_rate', 'liquidations'
    interval: str = '1m'  # For OHLCV: 1m, 5m, 1h, 1d, etc.
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 1000
    params: Dict[str, Any] = field(default_factory=dict)
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request"""
        key_parts = [
            self.source,
            self.symbol,
            self.data_type,
            self.interval,
            self.start_time.isoformat() if self.start_time else 'none',
            self.end_time.isoformat() if self.end_time else 'none',
            str(self.limit),
            json.dumps(self.params, sort_keys=True)
        ]
        key_string = '_'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class DataResponse:
    """Data response with metadata"""
    request: DataRequest
    data: Any
    timestamp: datetime
    source: str
    cache_hit: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request': {
                'source': self.request.source,
                'symbol': self.request.symbol,
                'data_type': self.request.data_type,
                'interval': self.request.interval
            },
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'cache_hit': self.cache_hit,
            'error': self.error,
            'data_size': len(self.data) if hasattr(self.data, '__len__') else 1
        }


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_second: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        
        # Tracking queues
        self.minute_queue: deque = deque(maxlen=requests_per_minute)
        self.second_queue: deque = deque(maxlen=requests_per_second)
        
        # Stats
        self.stats = {
            'total_requests': 0,
            'rate_limited': 0,
            'avg_wait_time': 0.0
        }
        
    def acquire(self) -> float:
        """Acquire permission to make request, returns wait time"""
        now = time.time()
        
        # Clean old entries
        while self.minute_queue and now - self.minute_queue[0] > 60:
            self.minute_queue.popleft()
        while self.second_queue and now - self.second_queue[0] > 1:
            self.second_queue.popleft()
        
        # Calculate wait time
        wait_time = 0.0
        
        # Check minute limit
        if len(self.minute_queue) >= self.requests_per_minute:
            oldest = self.minute_queue[0]
            wait_time = max(wait_time, 60 - (now - oldest))
            self.stats['rate_limited'] += 1
        
        # Check second limit
        if len(self.second_queue) >= self.requests_per_second:
            oldest = self.second_queue[0]
            wait_time = max(wait_time, 1 - (now - oldest))
            self.stats['rate_limited'] += 1
        
        # Wait if necessary
        if wait_time > 0:
            time.sleep(wait_time)
            now = time.time()  # Update time after waiting
        
        # Add new request times
        self.minute_queue.append(now)
        self.second_queue.append(now)
        self.stats['total_requests'] += 1
        
        # Update average wait time
        if self.stats['rate_limited'] > 0:
            self.stats['avg_wait_time'] = (
                (self.stats['avg_wait_time'] * (self.stats['rate_limited'] - 1) + wait_time) /
                self.stats['rate_limited']
            )
        
        return wait_time


class DataCache:
    """Intelligent data caching system"""
    
    def __init__(self, cache_dir: str = 'data/cache', max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        
        # Cache index
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_mb': 0.0
        }
        
        logger.info(f"DataCache initialized: {cache_dir}")
    
    def _load_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / 'index.json'
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
                
                # Calculate current size
                for key, metadata in self.cache_index.items():
                    file_path = self.cache_dir / f"{key}.pkl.gz"
                    if file_path.exists():
                        self.current_size_bytes += file_path.stat().st_size
                
            except Exception as e:
                logger.error(f"Error loading cache index: {e}")
                self.cache_index = {}
    
    def _save_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / 'index.json'
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if key not in self.cache_index:
            self.stats['misses'] += 1
            return None
        
        metadata = self.cache_index[key]
        file_path = self.cache_dir / f"{key}.pkl.gz"
        
        # Check expiry
        expiry_time = datetime.fromisoformat(metadata['expiry_time'])
        if datetime.now() > expiry_time:
            # Expired, remove from cache
            self._remove_key(key)
            self.stats['misses'] += 1
            return None
        
        try:
            # Load data
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            metadata['last_accessed'] = datetime.now().isoformat()
            metadata['access_count'] += 1
            self.cache_index[key] = metadata
            self._save_index()
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading cache data: {e}")
            self._remove_key(key)
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, data: Any, ttl_minutes: int = 60):
        """Store data in cache"""
        try:
            # Check cache size and evict if necessary
            self._evict_if_needed()
            
            # Store data
            file_path = self.cache_dir / f"{key}.pkl.gz"
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            file_size = file_path.stat().st_size
            self.current_size_bytes += file_size
            
            metadata = {
                'key': key,
                'size_bytes': file_size,
                'created_time': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'expiry_time': (datetime.now() + timedelta(minutes=ttl_minutes)).isoformat(),
                'access_count': 1,
                'ttl_minutes': ttl_minutes
            }
            
            self.cache_index[key] = metadata
            self._save_index()
            
            # Update stats
            self.stats['total_size_mb'] = self.current_size_bytes / (1024 * 1024)
            
            logger.debug(f"Cache set: {key} ({file_size/1024:.1f} KB)")
            
        except Exception as e:
            logger.error(f"Error setting cache data: {e}")
    
    def _evict_if_needed(self):
        """Evict old data if cache is full"""
        if self.current_size_bytes < self.max_size_bytes:
            return
        
        logger.info(f"Cache full ({self.current_size_bytes/(1024*1024):.1f} MB), evicting...")
        
        # Sort by last accessed time (oldest first)
        sorted_items = sorted(
            self.cache_index.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Evict until under limit
        while self.current_size_bytes > self.max_size_bytes * 0.9 and sorted_items:
            key, metadata = sorted_items.pop(0)
            self._remove_key(key)
            self.stats['evictions'] += 1
    
    def _remove_key(self, key: str):
        """Remove key from cache"""
        if key not in self.cache_index:
            return
        
        metadata = self.cache_index[key]
        file_path = self.cache_dir / f"{key}.pkl.gz"
        
        # Remove file
        if file_path.exists():
            file_size = file_path.stat().st_size
            self.current_size_bytes -= file_size
            file_path.unlink()
        
        # Remove from index
        del self.cache_index[key]
        self._save_index()
        
        logger.debug(f"Cache evicted: {key}")
    
    def clear(self):
        """Clear entire cache"""
        try:
            # Remove all cache files
            for file in self.cache_dir.glob("*.pkl.gz"):
                file.unlink()
            
            # Clear index
            self.cache_index.clear()
            self._save_index()
            
            # Reset stats
            self.current_size_bytes = 0
            self.stats['total_size_mb'] = 0.0
            
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.stats['hits'] + self.stats['misses'] > 0:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'current_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'items_count': len(self.cache_index)
        }


class DataFetcher:
    """Intelligent data fetcher with caching and rate limiting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.cache = DataCache(config.get('cache_dir', 'data/cache'))
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.get('requests_per_minute', 60),
            requests_per_second=config.get('requests_per_second', 10)
        )
        
        # API configurations
        self.api_configs = self._load_api_configs()
        
        # Session management
        self.http_session = None
        self.async_session = None
        
        # Request history
        self.request_history: deque = deque(maxlen=1000)
        
        # Data quality metrics
        self.quality_metrics = {
            'success_rate': 0.0,
            'avg_latency': 0.0,
            'total_requests': 0,
            'failed_requests': 0
        }
        
        logger.info("DataFetcher initialized")
    
    def _load_api_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load API configurations"""
        configs = {
            'binance': {
                'base_url': 'https://api.binance.com',
                'endpoints': {
                    'ohlcv': '/api/v3/klines',
                    'trades': '/api/v3/trades',
                    'orderbook': '/api/v3/depth',
                    'ticker': '/api/v3/ticker/24hr'
                },
                'rate_limit': 1200,  # requests per minute
                'requires_auth': False
            },
            'coinbase': {
                'base_url': 'https://api.pro.coinbase.com',
                'endpoints': {
                    'ohlcv': '/products/{symbol}/candles',
                    'trades': '/products/{symbol}/trades',
                    'orderbook': '/products/{symbol}/book'
                },
                'rate_limit': 10,  # requests per second
                'requires_auth': False
            },
            'kraken': {
                'base_url': 'https://api.kraken.com',
                'endpoints': {
                    'ohlcv': '/0/public/OHLC',
                    'trades': '/0/public/Trades',
                    'orderbook': '/0/public/Depth'
                },
                'rate_limit': 1,  # requests per second
                'requires_auth': False
            },
            'bybit': {
                'base_url': 'https://api.bybit.com',
                'endpoints': {
                    'ohlcv': '/v5/market/kline',
                    'trades': '/v5/market/recent-trade',
                    'orderbook': '/v5/market/orderbook'
                },
                'rate_limit': 50,  # requests per second
                'requires_auth': False
            },
            'alphavantage': {
                'base_url': 'https://www.alphavantage.co',
                'endpoints': {
                    'ohlcv': '/query',
                    'fundamentals': '/query'
                },
                'rate_limit': 5,  # requests per minute
                'requires_auth': True,
                'api_key': config.get('alphavantage_api_key')
            }
        }
        
        # Update with config overrides
        configs.update(self.config.get('api_configs', {}))
        return configs
    
    def fetch(self, request: DataRequest) -> DataResponse:
        """Fetch data with caching and retry logic"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = request.get_cache_key()
            cached_data = self.cache.get(cache_key)
            
            if cached_data is not None:
                logger.debug(f"Cache hit for {request.symbol} {request.data_type}")
                return DataResponse(
                    request=request,
                    data=cached_data,
                    timestamp=datetime.now(),
                    source=request.source,
                    cache_hit=True
                )
            
            # Apply rate limiting
            wait_time = self.rate_limiter.acquire()
            
            # Fetch from API
            data = self._fetch_from_api(request)
            
            # Cache the result
            if data is not None:
                # Determine TTL based on data type
                ttl = self._get_ttl_for_data_type(request.data_type)
                self.cache.set(cache_key, data, ttl)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Update quality metrics
            self._update_quality_metrics(success=True, latency=latency)
            
            # Create response
            response = DataResponse(
                request=request,
                data=data,
                timestamp=datetime.now(),
                source=request.source,
                cache_hit=False
            )
            
            # Log request
            self._log_request(request, response, latency, wait_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            
            # Update quality metrics
            self._update_quality_metrics(success=False)
            
            return DataResponse(
                request=request,
                data=None,
                timestamp=datetime.now(),
                source=request.source,
                cache_hit=False,
                error=str(e)
            )
    
    async def fetch_async(self, request: DataRequest) -> DataResponse:
        """Fetch data asynchronously"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = request.get_cache_key()
            cached_data = self.cache.get(cache_key)
            
            if cached_data is not None:
                logger.debug(f"Cache hit for {request.symbol} {request.data_type}")
                return DataResponse(
                    request=request,
                    data=cached_data,
                    timestamp=datetime.now(),
                    source=request.source,
                    cache_hit=True
                )
            
            # Apply rate limiting
            wait_time = self.rate_limiter.acquire()
            
            # Fetch from API
            data = await self._fetch_from_api_async(request)
            
            # Cache the result
            if data is not None:
                ttl = self._get_ttl_for_data_type(request.data_type)
                self.cache.set(cache_key, data, ttl)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Update quality metrics
            self._update_quality_metrics(success=True, latency=latency)
            
            # Create response
            response = DataResponse(
                request=request,
                data=data,
                timestamp=datetime.now(),
                source=request.source,
                cache_hit=False
            )
            
            # Log request
            self._log_request(request, response, latency, wait_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching data async: {e}")
            
            # Update quality metrics
            self._update_quality_metrics(success=False)
            
            return DataResponse(
                request=request,
                data=None,
                timestamp=datetime.now(),
                source=request.source,
                cache_hit=False,
                error=str(e)
            )
    
    def _fetch_from_api(self, request: DataRequest) -> Any:
        """Fetch data from API"""
        api_config = self.api_configs.get(request.source)
        if not api_config:
            raise ValueError(f"Unknown data source: {request.source}")
        
        endpoint_config = api_config['endpoints'].get(request.data_type)
        if not endpoint_config:
            raise ValueError(f"Unknown data type for {request.source}: {request.data_type}")
        
        # Build URL
        base_url = api_config['base_url']
        endpoint = endpoint_config.format(symbol=request.symbol)
        url = f"{base_url}{endpoint}"
        
        # Build parameters
        params = self._build_params(request, api_config)
        
        # Make request
        if self.http_session is None:
            self.http_session = requests.Session()
        
        response = self.http_session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Transform data based on data type
        return self._transform_data(data, request)
    
    async def _fetch_from_api_async(self, request: DataRequest) -> Any:
        """Fetch data from API asynchronously"""
        api_config = self.api_configs.get(request.source)
        if not api_config:
            raise ValueError(f"Unknown data source: {request.source}")
        
        endpoint_config = api_config['endpoints'].get(request.data_type)
        if not endpoint_config:
            raise ValueError(f"Unknown data type for {request.source}: {request.data_type}")
        
        # Build URL
        base_url = api_config['base_url']
        endpoint = endpoint_config.format(symbol=request.symbol)
        url = f"{base_url}{endpoint}"
        
        # Build parameters
        params = self._build_params(request, api_config)
        
        # Make request
        if self.async_session is None:
            self.async_session = aiohttp.ClientSession()
        
        async with self.async_session.get(url, params=params, timeout=30) as response:
            response.raise_for_status()
            data = await response.json()
        
        # Transform data
        return self._transform_data(data, request)
    
    def _build_params(self, request: DataRequest, api_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build API parameters based on source and data type"""
        params = {}
        
        if request.source == 'binance':
            if request.data_type == 'ohlcv':
                params = {
                    'symbol': request.symbol.replace('/', '').upper(),
                    'interval': request.interval,
                    'limit': request.limit
                }
                if request.start_time:
                    params['startTime'] = int(request.start_time.timestamp() * 1000)
                if request.end_time:
                    params['endTime'] = int(request.end_time.timestamp() * 1000)
        
        elif request.source == 'coinbase':
            if request.data_type == 'ohlcv':
                # Coinbase uses product IDs like 'BTC-USD'
                product_id = request.symbol.replace('/', '-')
                params = {
                    'granularity': self._interval_to_seconds(request.interval),
                    'start': request.start_time.isoformat() if request.start_time else None,
                    'end': request.end_time.isoformat() if request.end_time else None
                }
        
        elif request.source == 'kraken':
            if request.data_type == 'ohlcv':
                # Kraken uses pairs like 'XXBTZUSD'
                pair = request.symbol.replace('/', '').upper()
                params = {
                    'pair': pair,
                    'interval': self._interval_to_minutes(request.interval),
                    'since': int(request.start_time.timestamp()) if request.start_time else None
                }
        
        elif request.source == 'bybit':
            if request.data_type == 'ohlcv':
                # Bybit uses symbols like 'BTCUSD'
                symbol = request.symbol.replace('/', '').upper()
                params = {
                    'category': 'spot',
                    'symbol': symbol,
                    'interval': request.interval,
                    'limit': request.limit
                }
        
        elif request.source == 'alphavantage':
            if request.data_type == 'ohlcv':
                params = {
                    'function': 'TIME_SERIES_INTRADAY' if request.interval != '1d' else 'TIME_SERIES_DAILY',
                    'symbol': request.symbol.split('/')[0],  # Just the base currency
                    'interval': request.interval if request.interval != '1d' else None,
                    'outputsize': 'full' if request.limit > 100 else 'compact',
                    'apikey': api_config.get('api_key')
                }
        
        # Add custom params
        params.update(request.params)
        
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}
    
    def _transform_data(self, data: Any, request: DataRequest) -> Any:
        """Transform API response to standardized format"""
        if request.data_type == 'ohlcv':
            return self._transform_ohlcv_data(data, request.source)
        elif request.data_type == 'trades':
            return self._transform_trades_data(data, request.source)
        elif request.data_type == 'orderbook':
            return self._transform_orderbook_data(data, request.source)
        else:
            return data
    
    def _transform_ohlcv_data(self, data: Any, source: str) -> pd.DataFrame:
        """Transform OHLCV data to standardized DataFrame"""
        if source == 'binance':
            # Binance format: [time, open, high, low, close, volume, ...]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({'open': float, 'high': float, 'low': float, 
                           'close': float, 'volume': float})
        
        elif source == 'coinbase':
            # Coinbase format: [time, low, high, open, close, volume]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'low', 'high', 'open', 'close', 'volume'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({'open': float, 'high': float, 'low': float, 
                           'close': float, 'volume': float})
        
        elif source == 'kraken':
            # Kraken format: {pair: [[time, open, high, low, close, vwap, volume, count], ...]}
            pair = list(data['result'].keys())[0]
            ohlc_data = data['result'][pair]
            df = pd.DataFrame(ohlc_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({'open': float, 'high': float, 'low': float, 
                           'close': float, 'volume': float})
        
        else:
            # Default: assume it's already a DataFrame or list of lists
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def _transform_trades_data(self, data: Any, source: str) -> pd.DataFrame:
        """Transform trades data to standardized DataFrame"""
        # Implementation depends on API response format
        # This is a simplified version
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        return df
    
    def _transform_orderbook_data(self, data: Any, source: str) -> Dict[str, List]:
        """Transform orderbook data to standardized format"""
        if source == 'binance':
            return {
                'bids': [[float(price), float(quantity)] for price, quantity in data['bids']],
                'asks': [[float(price), float(quantity)] for price, quantity in data['asks']]
            }
        elif source == 'coinbase':
            return {
                'bids': [[float(bid[0]), float(bid[1])] for bid in data['bids']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in data['asks']]
            }
        else:
            return data
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            return 60  # Default 1 minute
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        else:
            return 1  # Default 1 minute
    
    def _get_ttl_for_data_type(self, data_type: str) -> int:
        """Get TTL in minutes for different data types"""
        ttl_config = {
            'ohlcv': 5,       # 5 minutes for OHLCV data
            'trades': 1,      # 1 minute for trades
            'orderbook': 0.1, # 6 seconds for orderbook
            'ticker': 1,      # 1 minute for ticker
            'funding_rate': 60, # 1 hour for funding rates
            'liquidations': 5   # 5 minutes for liquidations
        }
        return ttl_config.get(data_type, 5)
    
    def _update_quality_metrics(self, success: bool, latency: float = 0.0):
        """Update data quality metrics"""
        self.quality_metrics['total_requests'] += 1
        
        if success:
            # Update average latency
            total_requests = self.quality_metrics['total_requests']
            failed_requests = self.quality_metrics['failed_requests']
            successful_requests = total_requests - failed_requests - 1
            
            if successful_requests > 0:
                old_avg = self.quality_metrics['avg_latency']
                self.quality_metrics['avg_latency'] = (
                    (old_avg * successful_requests + latency) / (successful_requests + 1)
                )
            else:
                self.quality_metrics['avg_latency'] = latency
        else:
            self.quality_metrics['failed_requests'] += 1
        
        # Update success rate
        total = self.quality_metrics['total_requests']
        failed = self.quality_metrics['failed_requests']
        self.quality_metrics['success_rate'] = (total - failed) / total if total > 0 else 0.0
    
    def _log_request(self, request: DataRequest, response: DataResponse, 
                    latency: float, wait_time: float):
        """Log request details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': request.source,
            'symbol': request.symbol,
            'data_type': request.data_type,
            'cache_hit': response.cache_hit,
            'latency': latency,
            'wait_time': wait_time,
            'success': response.error is None,
            'data_size': len(response.data) if response.data is not None else 0
        }
        
        self.request_history.append(log_entry)
        
        if response.error:
            logger.warning(f"Request failed: {request.source} {request.symbol} "
                          f"{request.data_type} - {response.error}")
        else:
            logger.debug(f"Request completed: {request.source} {request.symbol} "
                        f"{request.data_type} in {latency:.3f}s")
    
    def batch_fetch(self, requests: List[DataRequest], 
                   max_concurrent: int = 5) -> List[DataResponse]:
        """Fetch multiple data requests with concurrency control"""
        results = []
        
        # Group by source for better rate limiting
        requests_by_source = defaultdict(list)
        for i, req in enumerate(requests):
            requests_by_source[req.source].append((i, req))
        
        # Process each source
        for source, source_requests in requests_by_source.items():
            # Process in batches
            for i in range(0, len(source_requests), max_concurrent):
                batch = source_requests[i:i + max_concurrent]
                
                # Fetch each in batch
                for idx, req in batch:
                    result = self.fetch(req)
                    results.append((idx, result))
                
                # Small delay between batches
                time.sleep(0.1)
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    async def batch_fetch_async(self, requests: List[DataRequest], 
                               max_concurrent: int = 10) -> List[DataResponse]:
        """Fetch multiple data requests asynchronously"""
        # Group by source
        requests_by_source = defaultdict(list)
        for i, req in enumerate(requests):
            requests_by_source[req.source].append((i, req))
        
        all_results = []
        
        # Process each source
        for source, source_requests in requests_by_source.items():
            # Create tasks
            tasks = []
            for idx, req in source_requests:
                task = asyncio.create_task(self.fetch_async(req))
                tasks.append((idx, task))
            
            # Run with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_fetch(idx, task):
                async with semaphore:
                    return idx, await task
            
            limited_tasks = [
                asyncio.create_task(limited_fetch(idx, task))
                for idx, task in tasks
            ]
            
            # Wait for all tasks
            source_results = await asyncio.gather(*limited_tasks)
            all_results.extend(source_results)
        
        # Sort by original order
        all_results.sort(key=lambda x: x[0])
        return [r[1] for r in all_results]
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get data quality report"""
        return {
            'quality_metrics': self.quality_metrics,
            'cache_stats': self.cache.get_stats(),
            'rate_limiter_stats': self.rate_limiter.stats,
            'recent_requests': list(self.request_history)[-10:]
        }
    
    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def close(self):
        """Clean up resources"""
        if self.http_session:
            self.http_session.close()
        
        if self.async_session:
            asyncio.run(self.async_session.close())
        
        logger.info("DataFetcher closed")


class MarketDataManager:
    """High-level market data manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fetcher = DataFetcher(config)
        
        # Data subscriptions
        self.subscriptions: Dict[str, List] = {}
        
        # Real-time data streams
        self.data_streams: Dict[str, Any] = {}
        
        # Historical data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        logger.info("MarketDataManager initialized")
    
    def get_ohlcv(self, symbol: str, interval: str = '1m', 
                  limit: int = 1000, source: str = 'binance') -> pd.DataFrame:
        """Get OHLCV data"""
        request = DataRequest(
            source=source,
            symbol=symbol,
            data_type='ohlcv',
            interval=interval,
            limit=limit
        )
        
        response = self.fetcher.fetch(request)
        
        if response.error:
            logger.error(f"Error fetching OHLCV: {response.error}")
            return pd.DataFrame()
        
        # Store in historical data
        key = f"{symbol}_{interval}"
        self.historical_data[key] = response.data
        
        return response.data
    
    def get_orderbook(self, symbol: str, source: str = 'binance') -> Dict[str, List]:
        """Get orderbook data"""
        request = DataRequest(
            source=source,
            symbol=symbol,
            data_type='orderbook',
            limit=100
        )
        
        response = self.fetcher.fetch(request)
        
        if response.error:
            logger.error(f"Error fetching orderbook: {response.error}")
            return {'bids': [], 'asks': []}
        
        return response.data
    
    def get_trades(self, symbol: str, limit: int = 100, 
                  source: str = 'binance') -> pd.DataFrame:
        """Get recent trades"""
        request = DataRequest(
            source=source,
            symbol=symbol,
            data_type='trades',
            limit=limit
        )
        
        response = self.fetcher.fetch(request)
        
        if response.error:
            logger.error(f"Error fetching trades: {response.error}")
            return pd.DataFrame()
        
        return response.data
    
    def subscribe(self, symbol: str, data_type: str, callback: callable):
        """Subscribe to real-time data"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        
        self.subscriptions[symbol].append({
            'data_type': data_type,
            'callback': callback,
            'last_update': datetime.now()
        })
        
        logger.info(f"Subscribed to {symbol} {data_type}")
    
    def unsubscribe(self, symbol: str, callback: callable):
        """Unsubscribe from real-time data"""
        if symbol in self.subscriptions:
            self.subscriptions[symbol] = [
                sub for sub in self.subscriptions[symbol]
                if sub['callback'] != callback
            ]
            
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
        
        logger.info(f"Unsubscribed from {symbol}")
    
    def update_all(self):
        """Update all subscribed data"""
        for symbol, subscriptions in self.subscriptions.items():
            for subscription in subscriptions:
                try:
                    # Fetch data
                    if subscription['data_type'] == 'ohlcv':
                        data = self.get_ohlcv(symbol)
                    elif subscription['data_type'] == 'orderbook':
                        data = self.get_orderbook(symbol)
                    elif subscription['data_type'] == 'trades':
                        data = self.get_trades(symbol)
                    else:
                        continue
                    
                    # Call callback
                    subscription['callback'](data)
                    subscription['last_update'] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error updating subscription for {symbol}: {e}")
    
    def get_historical_range(self, symbol: str, interval: str,
                            start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical data for a time range"""
        key = f"{symbol}_{interval}"
        
        if key in self.historical_data:
            df = self.historical_data[key]
            mask = (df.index >= start) & (df.index <= end)
            return df[mask]
        
        # Fetch if not available
        request = DataRequest(
            source='binance',  # Default source
            symbol=symbol,
            data_type='ohlcv',
            interval=interval,
            start_time=start,
            end_time=end,
            limit=1000
        )
        
        response = self.fetcher.fetch(request)
        
        if response.error:
            logger.error(f"Error fetching historical range: {response.error}")
            return pd.DataFrame()
        
        return response.data
    
    def export_data(self, symbol: str, interval: str, 
                   filepath: str, format: str = 'csv'):
        """Export data to file"""
        key = f"{symbol}_{interval}"
        
        if key not in self.historical_data:
            logger.error(f"No data available for {symbol} {interval}")
            return False
        
        df = self.historical_data[key]
        
        try:
            if format == 'csv':
                df.to_csv(filepath)
            elif format == 'parquet':
                df.to_parquet(filepath)
            elif format == 'json':
                df.to_json(filepath, orient='records')
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            'subscriptions': {
                symbol: len(subs) 
                for symbol, subs in self.subscriptions.items()
            },
            'historical_data_count': len(self.historical_data),
            'fetcher_quality': self.fetcher.get_quality_report()
        }