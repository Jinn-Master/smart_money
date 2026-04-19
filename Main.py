#!/usr/bin/env python3
"""
Institutional-Grade Smart Money Trading System
Follows: Wyckoff, VSA, Market Profile, Order Flow
"""

import asyncio
import signal
import sys
import logging
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

from core.market_microstructure import MarketMicrostructure
from core.order_flow import OrderFlowAnalyzer
from strategies.institutional.smart_money import SmartMoneyStrategy, SmartMoneySignal
from execution.smart_order_routing import SmartOrderRouter, ExecutionParameters, OrderType
from backtesting.event_driven import EventDrivenBacktester, BacktestConfig
from brokers import MultiBrokerExecution
from risk.advanced_risk import PortfolioManager
from monitoring.risk_dashboard import RiskDashboard

# Import data providers
try:
    from data_sources.alpha_vantage import AlphaVantageClient
    from data_sources.polygon import PolygonClient
    from data_sources.binance import BinanceClient
    DATA_SOURCES_AVAILABLE = True
except ImportError:
    DATA_SOURCES_AVAILABLE = False
    logger = logging.getLogger(__name__)

class InstitutionalTradingSystem:
    """Main institutional trading system"""
    
    def __init__(self, config_path: str = "config/institutional.yaml"):
        self.config = self._load_config(config_path)
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data clients
        self.data_clients = self._initialize_data_clients()
        
        # Core components
        self.strategy = SmartMoneyStrategy(config=self.config.get('strategy', {}))
        self.microstructure = MarketMicrostructure(config=self.config.get('microstructure', {}))
        self.order_flow = OrderFlowAnalyzer(config=self.config.get('order_flow', {}))
        
        # Execution
        self.broker = MultiBrokerExecution(self.config['brokers'])
        self.order_router = SmartOrderRouter(self.broker, config=self.config.get('execution', {}))
        
        # Risk management
        self.risk_manager = PortfolioManager(self.config['risk'])
        
        # Monitoring
        self.dashboard = RiskDashboard(self.config['monitoring'])
        
        # State
        self.active_signals: Dict[str, SmartMoneySignal] = {}
        self.positions = {}
        self.market_data_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info("Institutional Trading System initialized")
    
    def _setup_logging(self):
        """Configure logging system"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('file', 'logs/trading_system.log')
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger.info(f"Logging configured to {log_level} level")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set defaults for missing values
            config.setdefault('update_interval', 60)  # Default 60 seconds
            config.setdefault('symbols', [])
            config.setdefault('brokers', {})
            config.setdefault('risk', {})
            config.setdefault('monitoring', {})
            config.setdefault('data_sources', {})
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
    
    def _initialize_data_clients(self) -> Dict[str, Any]:
        """Initialize data source clients"""
        clients = {}
        data_config = self.config.get('data_sources', {})
        
        if DATA_SOURCES_AVAILABLE:
            # Initialize Alpha Vantage
            if 'alpha_vantage' in data_config:
                clients['alpha_vantage'] = AlphaVantageClient(
                    api_key=data_config['alpha_vantage']['api_key']
                )
            
            # Initialize Polygon
            if 'polygon' in data_config:
                clients['polygon'] = PolygonClient(
                    api_key=data_config['polygon']['api_key']
                )
            
            # Initialize Binance
            if 'binance' in data_config:
                clients['binance'] = BinanceClient(
                    api_key=data_config['binance']['api_key'],
                    api_secret=data_config['binance']['api_secret']
                )
        
        return clients
    
    async def initialize(self):
        """Initialize system components"""
        logger.info("Initializing institutional trading system...")
        
        # Initialize broker connections
        await self.broker.initialize()
        
        # Load historical data for calibration
        await self._load_historical_data()
        
        # Initialize risk manager
        await self.risk_manager.initialize()
        
        # Start monitoring dashboard
        self.dashboard.start()
        
        # Warm up strategy with recent data
        await self._warm_up_strategy()
        
        logger.info("System initialization complete")
        return True
    
    async def _load_historical_data(self):
        """Load historical data for strategy calibration"""
        logger.info("Loading historical data for calibration...")
        
        for symbol_config in self.config['symbols']:
            symbol = symbol_config['symbol']
            timeframe = symbol_config.get('timeframe', '1h')
            days_back = symbol_config.get('calibration_days', 30)
            
            try:
                # Try to load from cache first
                cache_key = f"{symbol}_{timeframe}"
                if cache_key in self.market_data_cache:
                    logger.info(f"Using cached data for {symbol}")
                    continue
                
                # Load historical data
                data = await self._fetch_historical_data(symbol, timeframe, days_back)
                if data is not None:
                    self.market_data_cache[cache_key] = data
                    logger.info(f"Loaded {len(data)} bars for {symbol}")
                    
                    # Warm up strategy indicators
                    self.strategy.warm_up(symbol, data)
                    
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
    
    async def _fetch_historical_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical price data from available sources"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try different data sources in order of preference
        for client_name, client in self.data_clients.items():
            try:
                if hasattr(client, 'get_historical_data'):
                    data = await client.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if data is not None and not data.empty:
                        logger.info(f"Fetched historical data for {symbol} from {client_name}")
                        return data
            except Exception as e:
                logger.warning(f"Failed to fetch data from {client_name}: {e}")
        
        # Fallback to file-based data
        data_path = f"data/{symbol.replace('/', '_')}_{timeframe}.csv"
        if Path(data_path).exists():
            try:
                data = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
                data = data.loc[start_date:end_date]
                if not data.empty:
                    logger.info(f"Loaded historical data for {symbol} from file")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load data from file: {e}")
        
        logger.error(f"No historical data available for {symbol}")
        return None
    
    async def _warm_up_strategy(self):
        """Warm up strategy with recent market data"""
        logger.info("Warming up strategy indicators...")
        
        for symbol_config in self.config['symbols']:
            symbol = symbol_config['symbol']
            cache_key = f"{symbol}_{symbol_config.get('timeframe', '1h')}"
            
            if cache_key in self.market_data_cache:
                data = self.market_data_cache[cache_key]
                try:
                    self.strategy.warm_up(symbol, data)
                    logger.info(f"Strategy warmed up for {symbol}")
                except Exception as e:
                    logger.error(f"Error warming up strategy for {symbol}: {e}")
    
    async def run(self, mode: str = "live"):
        """Run system in specified mode"""
        logger.info(f"Starting system in {mode} mode")
        
        if mode == "backtest":
            await self._run_backtest()
        elif mode == "paper":
            await self._run_paper_trading()
        elif mode == "live":
            await self._run_live_trading()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time market data"""
        try:
            market_data = {}
            
            # Fetch candles
            for client_name, client in self.data_clients.items():
                if hasattr(client, 'get_realtime_candles'):
                    candles = await client.get_realtime_candles(symbol)
                    if candles is not None:
                        market_data['candles'] = candles
                        break
            
            # Fetch order book (if supported)
            for client_name, client in self.data_clients.items():
                if hasattr(client, 'get_order_book'):
                    order_book = await client.get_order_book(symbol, depth=10)
                    if order_book is not None:
                        market_data['order_book'] = order_book
                        break
            
            # Fetch tick data (if supported)
            for client_name, client in self.data_clients.items():
                if hasattr(client, 'get_recent_ticks'):
                    ticks = await client.get_recent_ticks(symbol, limit=100)
                    if ticks is not None:
                        market_data['ticks'] = ticks
                        break
            
            if not market_data.get('candles'):
                logger.warning(f"No market data available for {symbol}")
                return None
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def _update_risk_metrics(self):
        """Update risk metrics and dashboard"""
        try:
            # Update portfolio metrics
            portfolio_value = await self.broker.get_portfolio_value()
            positions = await self.broker.get_positions()
            
            # Calculate current risk metrics
            risk_metrics = await self.risk_manager.calculate_metrics(
                portfolio_value=portfolio_value,
                positions=positions,
                market_data=self.market_data_cache
            )
            
            # Update dashboard
            self.dashboard.update_risk_metrics(risk_metrics)
            
            # Check for risk breaches
            breaches = await self.risk_manager.check_breaches(risk_metrics)
            if breaches:
                logger.warning(f"Risk breaches detected: {breaches}")
                await self._handle_risk_breaches(breaches)
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _handle_risk_breaches(self, breaches: List[str]):
        """Handle risk breaches by adjusting positions"""
        for breach in breaches:
            if 'drawdown' in breach.lower():
                # Reduce position sizes
                await self._reduce_exposure(0.5)  # Reduce by 50%
            elif 'var' in breach.lower():
                # Close riskiest positions
                await self._close_riskiest_positions()
            elif 'concentration' in breach.lower():
                # Diversify positions
                await self._diversify_positions()
    
    async def _reduce_exposure(self, reduction_factor: float):
        """Reduce overall exposure"""
        for symbol, position in list(self.positions.items()):
            new_size = position['size'] * reduction_factor
            if new_size < position['size']:  # Only reduce, don't increase
                await self._adjust_position(symbol, new_size)
    
    async def _adjust_position(self, symbol: str, new_size: float):
        """Adjust position size"""
        current_position = self.positions.get(symbol)
        if not current_position:
            return
        
        size_diff = new_size - current_position['size']
        if abs(size_diff) > 0:  # Only adjust if there's a difference
            side = 'SELL' if size_diff < 0 else 'BUY'
            
            execution_result = await self.order_router.execute_order(
                symbol=symbol,
                side=side,
                quantity=abs(size_diff),
                params=ExecutionParameters(
                    order_type=OrderType.MARKET,
                    urgency=0.8,
                    max_slippage=10.0
                )
            )
            
            if execution_result['success']:
                # Update position
                current_position['size'] = new_size
                logger.info(f"Adjusted position for {symbol} to {new_size}")
    
    async def _record_position(self, symbol: str, signal: SmartMoneySignal, execution_result: Dict):
        """Record new position"""
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.positions[symbol] = {
            'id': position_id,
            'symbol': symbol,
            'direction': signal.direction,
            'size': signal.position_size,
            'entry_price': execution_result['avg_price'],
            'entry_time': datetime.now(),
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'triggers': signal.triggers,
            'risk_score': signal.risk_score if hasattr(signal, 'risk_score') else 0.5
        }
        
        # Update active signals
        self.active_signals[symbol] = signal
        
        # Update performance metrics
        self.performance_metrics['total_trades'] += 1
        
        # Log position
        logger.info(f"Recorded new position: {symbol} {signal.direction} "
                   f"Size: {signal.position_size} @ {execution_result['avg_price']}")
    
    async def _close_position(self, symbol: str, position_id: str):
        """Close a position"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return
        
        position = self.positions[symbol]
        
        try:
            # Execute closing order
            execution_result = await self.order_router.execute_order(
                symbol=symbol,
                side='SELL' if position['direction'] == 'LONG' else 'BUY',
                quantity=position['size'],
                params=ExecutionParameters(
                    order_type=OrderType.MARKET,
                    urgency=0.9,
                    max_slippage=10.0
                )
            )
            
            if execution_result['success']:
                # Calculate P&L
                exit_price = execution_result['avg_price']
                pnl = self._calculate_pnl(position, exit_price)
                
                # Update performance metrics
                self.performance_metrics['total_pnl'] += pnl
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                
                # Remove position
                del self.positions[symbol]
                if symbol in self.active_signals:
                    del self.active_signals[symbol]
                
                logger.info(f"Closed position for {symbol}: P&L = {pnl:.2f}")
                
                # Update dashboard
                self.dashboard.update_trade_close(
                    symbol=symbol,
                    pnl=pnl,
                    exit_price=exit_price
                )
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L for a position"""
        if position['direction'] == 'LONG':
            return (exit_price - position['entry_price']) * position['size']
        else:  # SHORT
            return (position['entry_price'] - exit_price) * position['size']
    
    async def _load_backtest_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for backtesting"""
        historical_data = {}
        
        for symbol_config in self.config['symbols']:
            symbol = symbol_config['symbol']
            timeframe = symbol_config.get('timeframe', '1h')
            
            # Load data for backtest period
            data = await self._fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                days=365 * 3  # 3 years for backtesting
            )
            
            if data is not None:
                historical_data[symbol] = data
                logger.info(f"Loaded {len(data)} bars for {symbol} backtest")
        
        return historical_data
    
    def _generate_backtest_report(self, results: Dict):
        """Generate comprehensive backtest report"""
        report_path = "reports/backtest_report.html"
        
        # Create reports directory
        Path("reports").mkdir(exist_ok=True)
        
        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            <h2>Summary Statistics</h2>
            
            <div class="metric">
                <h3>Total Return: <span class="{ 'positive' if results['total_return'] > 0 else 'negative' }">
                    {results['total_return']:.2%}</span></h3>
            </div>
            
            <div class="metric">
                <h3>Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}</h3>
            </div>
            
            <div class="metric">
                <h3>Max Drawdown: <span class="negative">{results.get('max_drawdown', 0):.2%}</span></h3>
            </div>
            
            <h2>Trade Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{results.get('total_trades', 0)}</td></tr>
                <tr><td>Win Rate</td><td>{results.get('win_rate', 0):.2%}</td></tr>
                <tr><td>Average Win</td><td>{results.get('avg_win', 0):.2%}</td></tr>
                <tr><td>Average Loss</td><td>{results.get('avg_loss', 0):.2%}</td></tr>
                <tr><td>Profit Factor</td><td>{results.get('profit_factor', 0):.2f}</td></tr>
            </table>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Backtest report saved to {report_path}")
        
        # Also save JSON data for programmatic access
        json_path = "reports/backtest_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("Emergency shutdown initiated")
        
        try:
            # Cancel all pending orders
            await self.broker.cancel_all_orders()
            
            # Close all positions immediately
            for symbol in list(self.positions.keys()):
                await self._close_position(symbol, self.positions[symbol]['id'])
            
            # Flush all logs
            logging.shutdown()
            
            # Send alert
            await self._send_emergency_alert()
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
    
    async def _send_emergency_alert(self):
        """Send emergency alert"""
        # Implement alerting (email, Slack, SMS, etc.)
        alert_message = f"EMERGENCY SHUTDOWN - Trading System {datetime.now()}"
        logger.critical(alert_message)
        
        # Here you would integrate with your alerting system
        # Example: await self.dashboard.send_alert(alert_message)
    
    async def _run_live_trading(self):
        """Run live trading with real money"""
        self.running = True
        
        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        try:
            while self.running:
                # Process each symbol
                for symbol_config in self.config['symbols']:
                    symbol = symbol_config['symbol']
                    
                    # 1. Fetch market data
                    market_data = await self._fetch_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # 2. Analyze for smart money signals
                    signal = self.strategy.analyze_market(
                        symbol=symbol,
                        candles=market_data['candles'],
                        order_book=market_data.get('order_book'),
                        ticks=market_data.get('ticks')
                    )
                    
                    # 3. Risk approval
                    if signal.direction != "NEUTRAL":
                        approved = await self.risk_manager.approve_trade(signal)
                        
                        if approved:
                            # 4. Calculate position size
                            position_size = self.risk_manager.calculate_position_size(signal)
                            signal.position_size = position_size
                            
                            # 5. Execute with smart routing
                            execution_params = ExecutionParameters(
                                order_type=OrderType.ICEBERG if position_size > 10000 else OrderType.VWAP,
                                urgency=0.7 if 'liquidity_grab' in signal.triggers else 0.3,
                                max_slippage=5.0,  # 5 bps
                                max_participation=0.1,  # 10% of volume
                                schedule=[]
                            )
                            
                            execution_result = await self.order_router.execute_order(
                                symbol=symbol,
                                side='BUY' if signal.direction == 'LONG' else 'SELL',
                                quantity=position_size,
                                params=execution_params
                            )
                            
                            if execution_result['success']:
                                # 6. Record position
                                await self._record_position(symbol, signal, execution_result)
                                
                                # 7. Update dashboard
                                self.dashboard.update_trade(
                                    symbol=symbol,
                                    direction=signal.direction,
                                    size=position_size,
                                    price=execution_result['avg_price']
                                )
                
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Check for position exits
                await self._check_position_exits()
                
                # Sleep based on timeframe
                await asyncio.sleep(self.config['update_interval'])
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self._emergency_shutdown()
        finally:
            await self.shutdown()
    
    async def _check_position_exits(self):
        """Check if any positions should be exited"""
        for symbol, position in list(self.positions.items()):
            # Get current price
            market_data = await self._fetch_market_data(symbol)
            if not market_data or 'candles' not in market_data:
                continue
            
            current_price = market_data['candles'].iloc[-1]['close']
            
            # Check stop loss
            if position['direction'] == 'LONG' and current_price <= position['stop_loss']:
                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                await self._close_position(symbol, position['id'])
            elif position['direction'] == 'SHORT' and current_price >= position['stop_loss']:
                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                await self._close_position(symbol, position['id'])
            
            # Check take profit
            if position['direction'] == 'LONG' and current_price >= position['take_profit']:
                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                await self._close_position(symbol, position['id'])
            elif position['direction'] == 'SHORT' and current_price <= position['take_profit']:
                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                await self._close_position(symbol, position['id'])
    
    async def _run_backtest(self):
        """Run comprehensive backtest"""
        logger.info("Starting backtest...")
        
        # Load historical data
        historical_data = await self._load_backtest_data()
        
        # Configure backtester
        backtest_config = BacktestConfig(
            initial_capital=self.config['risk']['initial_capital'],
            commission_bps=1.0,
            slippage_bps=0.5,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        backtester = EventDrivenBacktester(backtest_config)
        
        # Define strategy wrapper for backtesting
        def backtest_strategy(prices, timestamp):
            signals = []
            
            for symbol in historical_data.keys():
                # Get data up to current timestamp
                data_up_to_now = historical_data[symbol].loc[:timestamp]
                
                if len(data_up_to_now) > 100:  # Need enough data
                    signal = self.strategy.analyze_market(
                        symbol=symbol,
                        candles=data_up_to_now,
                        order_book=None,
                        ticks=None
                    )
                    
                    if signal.direction != "NEUTRAL":
                        # Convert to backtest signal format
                        signals.append({
                            'symbol': symbol,
                            'direction': signal.direction,
                            'quantity': signal.position_size,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'metadata': {'triggers': signal.triggers}
                        })
            
            return signals
        
        # Run backtest
        results = backtester.run(backtest_strategy, historical_data)
        
        # Generate report
        self._generate_backtest_report(results)
        
        logger.info("Backtest complete")
    
    async def _run_paper_trading(self):
        """Run paper trading (simulated execution)"""
        logger.info("Starting paper trading...")
        
        # Use simulated broker
        from brokers.simulated import SimulatedBroker
        self.broker = SimulatedBroker(
            initial_capital=self.config['risk']['initial_capital']
        )
        
        # Run similar to live but with simulated broker
        await self._run_live_trading()
        
        logger.info("Paper trading complete")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating shutdown...")
        self.running = False
        
        # Close all positions
        for symbol, position in list(self.positions.items()):
            await self._close_position(symbol, position['id'])
        
        # Shutdown components
        await self.broker.shutdown()
        self.dashboard.shutdown()
        
        # Generate final report
        self._generate_performance_report()
        
        logger.info("Shutdown complete")
    
    def _generate_performance_report(self):
        """Generate final performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance': self.performance_metrics,
            'final_positions': len(self.positions),
            'active_signals': len(self.active_signals)
        }
        
        report_path = f"reports/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")

async def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Institutional Trading System")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], 
                       default="backtest", help="Operation mode")
    parser.add_argument("--config", default="config/institutional.yaml", 
                       help="Config file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run system
    system = InstitutionalTradingSystem(args.config)
    
    try:
        if await system.initialize():
            await system.run(args.mode)
        else:
            logger.error("Failed to initialize system")
            return 1
    except Exception as e:
        logger.critical(f"System crashed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))