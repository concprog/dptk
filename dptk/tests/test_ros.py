import pytest
from unittest.mock import patch, MagicMock
from dptk.stream import Stream, run

def test_run_logic():
    # Test that run correctly identifies roots and waits for them
    s1 = Stream()
    s2 = s1.pipe(lambda x: x)
    
    mock_worker = MagicMock()
    mock_worker.is_alive.side_effect = [True, False]
    s1._worker = mock_worker
    
    # Running should join the worker and not hang
    run(s2)
    
    assert mock_worker.is_alive.call_count == 2
    mock_worker.join.assert_called_once()
