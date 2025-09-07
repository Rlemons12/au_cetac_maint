# modules/initial_setup/audit_system_setup.py

import os
import sys
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    # ✅ Import the stub (or real) audit system from modules.initial_setup
    from modules.initial_setup.audit_system import (
        AuditManager, PostgreSQLAuditTriggers, AuditLog,
        setup_complete_audit_system, AuditMixin
    )
    AUDIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import audit system stub: {e}")
    AUDIT_AVAILABLE = False


def setup_basic_audit_system():
    """Fallback audit setup when full audit system is unavailable"""
    print("⚠️ Audit system not available, skipping setup.")
    return False


def main():
    if AUDIT_AVAILABLE:
        try:
            print("✅ Running audit system setup (stub).")
            setup_complete_audit_system()
            return True
        except Exception as e:
            print(f"❌ Audit system setup failed: {e}")
            return False
    else:
        return setup_basic_audit_system()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
