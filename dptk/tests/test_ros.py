import pytest
from unittest.mock import patch, MagicMock
from dptk.stream import Stream, wait_till_complete


def test_run_logic():
    # Test that run correctly identifies roots and waits for them
    s1 = Stream()
    s2 = s1.pipe(lambda x: x)

    mock_worker = MagicMock()
    mock_worker.is_alive.return_value = True
    s1._worker = mock_worker

    # Running should join the worker and not hang
    with patch("dptk.stream.time.sleep", side_effect=KeyboardInterrupt):
        wait_till_complete(s2)

    mock_worker.join.assert_called_once()
