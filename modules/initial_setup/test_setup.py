import os
import subprocess
import sys
import types
import pytest
import importlib.util

# --- Stub out modules that have side-effects on import ---
# modules.emtacdb.emtac_revision_control_db should export RevisionControlBase and MainBase
rcm = types.ModuleType('modules.emtacdb.emtac_revision_control_db')
rcm.RevisionControlBase = type('RevisionControlBase', (), {})
rcm.MainBase = type('MainBase', (), {})
sys.modules['modules.emtacdb.emtac_revision_control_db'] = rcm

# Stub snapshot_utils module to prevent import failures
snap = types.ModuleType('modules.emtacdb.utlity.revision_database.snapshot_utils')
sys.modules['modules.emtacdb.utlity.revision_database.snapshot_utils'] = snap

# Fixtures to provide an isolated mock_setup directory
@pytest.fixture
def mock_root(tmp_path):
    root = tmp_path / 'mock_setup'
    return str(root)

# Dynamically load the setup module from its file location
module_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'emtacdb_initial_setup.py')
)
spec = importlib.util.spec_from_file_location("setup_module", module_path)
setup = importlib.util.module_from_spec(spec)
# Prevent executing main() on import (script should guard under __main__)
spec.loader.exec_module(setup)

class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)
    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)
    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)

@pytest.fixture(autouse=True)
def dummy_logger(monkeypatch):
    dummy = DummyLogger()
    # Patch both logger names
    monkeypatch.setattr(setup, 'initializer_logger', dummy, raising=False)
    monkeypatch.setattr(setup, 'logger', dummy, raising=False)
    return dummy

@pytest.fixture(autouse=True)
def fake_subprocess(monkeypatch):
    calls = []
    def fake_run(cmd, check=False):
        calls.append(cmd)
        class Result:
            returncode = 0
        return Result()
    monkeypatch.setattr(subprocess, 'run', fake_run)
    return calls

# Test requirements installation using isolated root
def test_check_and_install_requirements_installs(mock_root, dummy_logger, fake_subprocess):
    root_dir = mock_root
    os.makedirs(root_dir, exist_ok=True)
    req_path = os.path.join(root_dir, 'requirements.txt')
    open(req_path, 'w').close()

    setup.check_and_install_requirements(root_dir=root_dir)

    assert fake_subprocess, "pip install should have been invoked"
    assert any('-r' in ' '.join(cmd) for cmd in fake_subprocess), "requirements.txt flag missing"
    assert any('dependencies installed' in msg for msg in dummy_logger.infos)

# Test skipping when no requirements
def test_check_and_install_requirements_skips(mock_root, dummy_logger, fake_subprocess):
    setup.check_and_install_requirements(root_dir=mock_root)
    assert not fake_subprocess
    assert dummy_logger.warnings, "Expected warning for missing requirements"

# Test directory creation under isolated root
def test_create_directories(mock_root, dummy_logger):
    setup.create_directories(root_dir=mock_root)
    expected = [
        'DB_IMAGES', 'logs', 'DB_LOADSHEET_BOMS', 'DB_DOC',
        'db_backup', 'DB_LOADSHEETS', 'DB_LOADSHEETS_BACKUP',
        'PDF_FILES', 'PPT_FILES'
    ]
    for d in expected:
        full = os.path.join(mock_root, d)
        assert os.path.isdir(full), f"{full} not created"
    assert dummy_logger.infos

# Dummy classes for database tests
class DummyInspector:
    def __init__(self, tables): self._tables = tables
    def get_table_names(self): return self._tables
class DummyEngine: pass
class DummyBase:
    def __init__(self): self.created = False; self.metadata = self
    def create_all(self, engine): self.created = True

# Test database creation when no tables exist
def test_check_and_create_database_creates(monkeypatch, mock_root, dummy_logger):
    class DummyConfig:
        def __init__(self):
            self.main_engine = DummyEngine()
            self.revision_control_engine = DummyEngine()
            self.MainBase = DummyBase()
            self.RevisionControlBase = DummyBase()
    monkeypatch.setattr(setup, 'DatabaseConfig', DummyConfig)
    monkeypatch.setattr(setup, 'sa_inspect', lambda eng: DummyInspector([]))

    setup.check_and_create_database(root_dir=mock_root)
    # If no exception, logic executed

# Test skipping database creation when tables exist
def test_check_and_create_database_skips(monkeypatch, mock_root, dummy_logger):
    class DummyConfig2(DummyConfig): pass
    monkeypatch.setattr(setup, 'DatabaseConfig', DummyConfig2)
    monkeypatch.setattr(setup, 'sa_inspect', lambda eng: DummyInspector(['tbl']))

    setup.check_and_create_database(root_dir=mock_root)
    # If no exception, skip logic executed

# Test running setup scripts under isolated root
def test_run_setup_scripts(mock_root, fake_subprocess, dummy_logger):
    os.makedirs(mock_root, exist_ok=True)
    scripts = [
        'load_equipment_relationships_table_data.py',
        'initial_admin.py',
        'load_parts_sheet.py',
        'load_active_drawing_list.py',
        'load_image_folder.py',
        'load_bom_loadsheet.py',
    ]
    for s in scripts: open(os.path.join(mock_root, s), 'w').close()

    setup.run_setup_scripts(root_dir=mock_root)
    assert len(fake_subprocess) == len(scripts)
    for cmd, s in zip(fake_subprocess, scripts):
        assert cmd[-1].endswith(s)
