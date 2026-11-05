import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.database_manager import DatabaseManager  # noqa: E402
from modules.inventory_manager import InventoryManager  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    return DatabaseManager(tmp_path / "test.db")


@pytest.fixture()
def inventory(db):
    return InventoryManager(db)
