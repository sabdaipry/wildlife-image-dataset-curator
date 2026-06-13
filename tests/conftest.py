import sys
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """QApplication singleton shared across the entire test session."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app
