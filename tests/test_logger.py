# tests/test_logger.py
import sys
import os
import tempfile
import logging
import pytest
sys.path.insert(0, '.')
from project_utils.logger import PipelineLogger

def test_logger_creation():
    """Test basic logger creation."""
    logger = PipelineLogger(name="test_logger", level="INFO")
    assert logger.get_level() == "INFO"

def test_logger_level_validation():
    """Test that invalid log levels raise ValueError."""
    # Valid levels should work
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logger = PipelineLogger(name=f"test_{level}", level=level)
        assert logger.get_level() == level
    
    # Invalid levels should raise ValueError
    with pytest.raises(ValueError, match="Invalid log level"):
        PipelineLogger(name="test_invalid", level="INVALID")
    
    with pytest.raises(ValueError, match="Invalid log level"):
        PipelineLogger(name="test_invalid2", level="debugging")

def test_logger_methods():
    """Test all logging methods."""
    logger = PipelineLogger(name="test_methods", level="DEBUG")
    
    # Test each method (they shouldn't raise exceptions)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Test exception method
    try:
        raise ValueError("Test exception")
    except ValueError:
        logger.exception("Exception occurred")

def test_logger_file_logging():
    """Test logging to a file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    try:
        logger = PipelineLogger(
            name="test_file_logger", 
            level="INFO",
            log_file=log_file
        )
        
        # Log a message
        test_message = "Test file logging message"
        logger.info(test_message)
        
        # Force handlers to flush
        for handler in logger.logger.handlers:
            handler.flush()
        
        # Check file contents
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content
            assert "INFO" in content
            assert "test_file_logger" in content
    finally:
        # Clean up
        if os.path.exists(log_file):
            os.remove(log_file)

def test_handler_deduplication():
    """Test that duplicate handlers are not added."""
    logger1 = PipelineLogger(name="test_dedup", level="INFO")
    handler_count_before = len(logger1.logger.handlers)
    
    # Create another logger with same name
    logger2 = PipelineLogger(name="test_dedup", level="INFO")
    handler_count_after = len(logger2.logger.handlers)
    
    # Should have same number of handlers (no duplicates added)
    assert handler_count_after == handler_count_before
    
    # Create with different name should add new handlers
    logger3 = PipelineLogger(name="test_dedup_different", level="INFO")
    assert len(logger3.logger.handlers) >= handler_count_before

def test_formatter_parameter():
    """Test custom formatter parameter."""
    custom_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    logger = PipelineLogger(
        name="test_formatter",
        level="INFO",
        formatter=custom_formatter
    )
    
    # Check that at least one handler uses our formatter
    for handler in logger.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            assert handler.formatter == custom_formatter
            break

def test_logger_level_changes():
    """Test that different levels are respected."""
    # Test DEBUG level
    logger_debug = PipelineLogger(name="test_level_debug", level="DEBUG")
    assert logger_debug.get_level() == "DEBUG"
    
    # Test WARNING level  
    logger_warning = PipelineLogger(name="test_level_warning", level="WARNING")
    assert logger_warning.get_level() == "WARNING"
    
    # Test ERROR level
    logger_error = PipelineLogger(name="test_level_error", level="ERROR")
    assert logger_error.get_level() == "ERROR"
    
    # Test CRITICAL level
    logger_critical = PipelineLogger(name="test_level_critical", level="CRITICAL")
    assert logger_critical.get_level() == "CRITICAL"

def test_backward_compatibility():
    """Test backward compatibility with original interface."""
    # Original parameters should still work
    logger1 = PipelineLogger(name="test_compat1", level="INFO")
    assert logger1.get_level() == "INFO"
    
    logger2 = PipelineLogger(name="test_compat2", level="INFO", log_file=None)
    assert logger2.get_level() == "INFO"
    
    # Original methods should still work
    logger2.info("Test message")
    logger2.warning("Test warning")
    logger2.error("Test error")
