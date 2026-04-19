"""
Advanced Logging System
Structured logging with different levels, handlers, and monitoring
"""

import logging
import logging.handlers
from logging import Logger
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import traceback
import inspect
import hashlib
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import time
import coloredlogs
import verboselogs

# Custom log levels
SUCCESS = 25  # Between INFO and WARNING
AUDIT = 35    # Between WARNING and ERROR

# Add custom levels
logging.addLevelName(SUCCESS, 'SUCCESS')
logging.addLevelName(AUDIT, 'AUDIT')


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_no: int
    thread_id: int
    process_id: int
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    extra: Dict[str, Any] = None
    correlation_id: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class StructuredFormatter(logging.Formatter):
    """Structured log formatter"""
    
    def __init__(self, 
                 fmt: Optional[str] = None, 
                 datefmt: Optional[str] = None,
                 style: str = '%'):
        super().__init__(fmt, datefmt, style)
        self.default_fmt = fmt
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record"""
        # Create structured data
        structured_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process,
            'thread_name': record.threadName,
            'process_name': record.processName
        }
        
        # Add exception info if present
        if record.exc_info:
            structured_data['exception'] = record.exc_text
            structured_data['stack_trace'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            structured_data.update(record.extra)
        
        if hasattr(record, 'correlation_id'):
            structured_data['correlation_id'] = record.correlation_id
        
        # Format based on style
        if self.default_fmt:
            # Use traditional formatting
            return super().format(record)
        else:
            # Return JSON
            return json.dumps(structured_data, default=str)


class LogMonitor:
    """Real-time log monitor"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.log_buffer: List[LogEntry] = []
        self.subscribers: List[callable] = []
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_logs': 0,
            'by_level': defaultdict(int),
            'by_logger': defaultdict(int),
            'errors_last_hour': 0,
            'warnings_last_hour': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate': 10,  # errors per minute
            'warning_rate': 50,  # warnings per minute
            'memory_usage': 80  # percent
        }
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def add_entry(self, entry: LogEntry):
        """Add log entry to monitor"""
        with self._lock:
            # Add to buffer
            self.log_buffer.append(entry)
            if len(self.log_buffer) > self.buffer_size:
                self.log_buffer.pop(0)
            
            # Update statistics
            self.stats['total_logs'] += 1
            self.stats['by_level'][entry.level] += 1
            self.stats['by_logger'][entry.logger_name] += 1
            
            # Check for alerts
            self._check_alerts(entry)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(entry)
                except Exception as e:
                    print(f"Error in log subscriber: {e}", file=sys.stderr)
    
    def subscribe(self, callback: callable):
        """Subscribe to log updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: callable):
        """Unsubscribe from log updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _check_alerts(self, entry: LogEntry):
        """Check for alert conditions"""
        # Count recent errors and warnings
        if entry.level == 'ERROR':
            self.stats['errors_last_hour'] += 1
        elif entry.level == 'WARNING':
            self.stats['warnings_last_hour'] += 1
        
        # Check error rate
        if self.stats['errors_last_hour'] > self.alert_thresholds['error_rate'] * 60:
            # High error rate alert
            self._send_alert(f"High error rate detected: {self.stats['errors_last_hour']} errors in last hour")
        
        # Check for critical errors
        if entry.level == 'CRITICAL':
            self._send_alert(f"CRITICAL error: {entry.message}")
    
    def _send_alert(self, message: str):
        """Send alert (could be email, slack, etc.)"""
        # In production, this would send actual alerts
        print(f"ALERT: {message}", file=sys.stderr)
    
    def start(self):
        """Start log monitor"""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop log monitor"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Monitor loop for cleanup and stats"""
        while self._running:
            try:
                # Clean old stats every hour
                self.stats['errors_last_hour'] = 0
                self.stats['warnings_last_hour'] = 0
                
                time.sleep(3600)  # Sleep for 1 hour
                
            except Exception as e:
                print(f"Error in log monitor loop: {e}", file=sys.stderr)
                time.sleep(60)
    
    def get_recent_logs(self, level: Optional[str] = None, 
                       limit: int = 100) -> List[LogEntry]:
        """Get recent logs with optional filtering"""
        with self._lock:
            if level:
                filtered = [entry for entry in self.log_buffer if entry.level == level]
                return filtered[-limit:]
            else:
                return self.log_buffer[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['buffer_size'] = len(self.log_buffer)
            stats['subscriber_count'] = len(self.subscribers)
            return stats


class AuditLogger:
    """Audit logging for compliance and security"""
    
    def __init__(self, log_file: str = 'logs/audit.log'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create audit logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(AUDIT)
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        
        # Structured formatter
        formatter = StructuredFormatter()
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def log_event(self, 
                  event_type: str,
                  user: str,
                  action: str,
                  resource: str,
                  status: str,
                  details: Dict[str, Any] = None,
                  correlation_id: str = None):
        """Log audit event"""
        extra = {
            'event_type': event_type,
            'user': user,
            'action': action,
            'resource': resource,
            'status': status,
            'details': details or {}
        }
        
        if correlation_id:
            extra['correlation_id'] = correlation_id
        
        # Get caller info
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame).__name__ if inspect.getmodule(frame) else 'unknown'
        function = frame.f_code.co_name
        line_no = frame.f_lineno
        
        # Create log record
        record = self.logger.makeRecord(
            name='audit',
            level=AUDIT,
            fn=module,
            lno=line_no,
            msg=f"{user} {action} {resource} - {status}",
            args=(),
            exc_info=None,
            extra=extra,
            func=function
        )
        
        # Add correlation ID
        if correlation_id:
            record.correlation_id = correlation_id
        
        self.logger.handle(record)


class LoggerFactory:
    """Factory for creating configured loggers"""
    
    _instances: Dict[str, Logger] = {}
    _monitor: Optional[LogMonitor] = None
    _audit_logger: Optional[AuditLogger] = None
    
    @classmethod
    def setup_logging(cls, 
                      config: Dict[str, Any],
                      enable_monitor: bool = True,
                      enable_audit: bool = True):
        """Setup global logging configuration"""
        
        # Get configuration
        log_level = config.get('log_level', 'INFO')
        log_dir = Path(config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup colored logs for console
        coloredlogs.install(
            level=log_level,
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            field_styles={
                'asctime': {'color': 'green'},
                'levelname': {'color': 'magenta', 'bold': True},
                'name': {'color': 'blue'}
            }
        )
        
        # Setup file logging
        file_formatter = StructuredFormatter()
        
        # Main log file
        main_file = log_dir / 'main.log'
        file_handler = logging.handlers.RotatingFileHandler(
            main_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Error log file
        error_file = log_dir / 'error.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.WARNING)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        
        # Remove default handlers to avoid duplicate logs
        root_logger.handlers = [h for h in root_logger.handlers 
                               if not isinstance(h, logging.StreamHandler)]
        
        # Add console handler with colored logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        colored_formatter = coloredlogs.ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        console_handler.setFormatter(colored_formatter)
        root_logger.addHandler(console_handler)
        
        # Initialize monitor
        if enable_monitor:
            cls._monitor = LogMonitor()
            cls._monitor.start()
            
            # Add monitor handler
            monitor_handler = MonitorHandler(cls._monitor)
            monitor_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(monitor_handler)
        
        # Initialize audit logger
        if enable_audit:
            audit_file = log_dir / 'audit.log'
            cls._audit_logger = AuditLogger(str(audit_file))
        
        logger = cls.get_logger('LoggerFactory')
        logger.info(f"Logging setup complete. Level: {log_level}, Directory: {log_dir}")
    
    @classmethod
    def get_logger(cls, name: str, correlation_id: str = None) -> Logger:
        """Get configured logger instance"""
        if name not in cls._instances:
            logger = verboselogs.VerboseLogger(name)
            
            # Add custom methods
            logger.success = lambda msg, *args, **kwargs: logger.log(
                SUCCESS, msg, *args, **kwargs
            )
            logger.audit = lambda msg, *args, **kwargs: logger.log(
                AUDIT, msg, *args, **kwargs
            )
            
            # Store correlation ID in logger
            if correlation_id:
                logger.correlation_id = correlation_id
            
            cls._instances[name] = logger
        
        return cls._instances[name]
    
    @classmethod
    def get_monitor(cls) -> Optional[LogMonitor]:
        """Get log monitor instance"""
        return cls._monitor
    
    @classmethod
    def get_audit_logger(cls) -> Optional[AuditLogger]:
        """Get audit logger instance"""
        return cls._audit_logger
    
    @classmethod
    def shutdown(cls):
        """Shutdown logging system"""
        if cls._monitor:
            cls._monitor.stop()
        
        logging.shutdown()
        
        logger = cls.get_logger('LoggerFactory')
        logger.info("Logging system shutdown")


class MonitorHandler(logging.Handler):
    """Custom handler for log monitor"""
    
    def __init__(self, monitor: LogMonitor):
        super().__init__()
        self.monitor = monitor
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to monitor"""
        try:
            # Create log entry
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_no=record.lineno,
                thread_id=record.thread,
                process_id=record.process,
                exception=record.exc_text if record.exc_info else None,
                stack_trace=self.formatException(record.exc_info) if record.exc_info else None,
                extra=getattr(record, 'extra', None),
                correlation_id=getattr(record, 'correlation_id', None)
            )
            
            # Add to monitor
            self.monitor.add_entry(entry)
            
        except Exception as e:
            print(f"Error in MonitorHandler: {e}", file=sys.stderr)


class CorrelationContext:
    """Context manager for correlation IDs"""
    
    def __init__(self, correlation_id: str = None):
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.old_correlation_id = None
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"corr_{timestamp}_{random_part}"
    
    def __enter__(self):
        """Enter context"""
        frame = inspect.currentframe().f_back
        
        # Find logger in calling scope
        for var_name, var_value in frame.f_locals.items():
            if isinstance(var_value, logging.Logger):
                self.old_correlation_id = getattr(var_value, 'correlation_id', None)
                var_value.correlation_id = self.correlation_id
                break
        
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        frame = inspect.currentframe().f_back
        
        # Restore old correlation ID
        for var_name, var_value in frame.f_locals.items():
            if isinstance(var_value, logging.Logger):
                var_value.correlation_id = self.old_correlation_id
                break


# Convenience functions
def setup_logging(config: Dict[str, Any] = None):
    """Setup logging with default configuration"""
    default_config = {
        'log_level': 'INFO',
        'log_dir': 'logs',
        'enable_monitor': True,
        'enable_audit': True
    }
    
    if config:
        default_config.update(config)
    
    LoggerFactory.setup_logging(default_config)


def get_logger(name: str, correlation_id: str = None) -> Logger:
    """Get logger instance"""
    return LoggerFactory.get_logger(name, correlation_id)


def log_success(logger: Logger, message: str, *args, **kwargs):
    """Log success message"""
    logger.log(SUCCESS, message, *args, **kwargs)


def log_audit(logger: Logger, message: str, *args, **kwargs):
    """Log audit message"""
    logger.log(AUDIT, message, *args, **kwargs)


def get_log_monitor() -> Optional[LogMonitor]:
    """Get log monitor"""
    return LoggerFactory.get_monitor()


def get_audit_logger() -> Optional[AuditLogger]:
    """Get audit logger"""
    return LoggerFactory.get_audit_logger()


def shutdown_logging():
    """Shutdown logging system"""
    LoggerFactory.shutdown()


# Example usage
if __name__ == '__main__':
    # Setup logging
    setup_logging({
        'log_level': 'DEBUG',
        'log_dir': 'test_logs'
    })
    
    # Get logger
    logger = get_logger(__name__)
    
    # Log with different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.success("Success message")  # Custom level
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    logger.audit("Audit message")  # Custom level
    
    # Log with correlation ID
    with CorrelationContext() as corr_id:
        logger.info(f"Message with correlation ID: {corr_id}")
    
    # Get monitor stats
    monitor = get_log_monitor()
    if monitor:
        print("Recent logs:")
        for entry in monitor.get_recent_logs(limit=5):
            print(f"{entry.timestamp} [{entry.level}] {entry.message}")
        
        print("\nStats:", monitor.get_stats())
    
    # Shutdown
    shutdown_logging()