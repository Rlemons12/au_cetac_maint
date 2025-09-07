#!/usr/bin/env python3
"""
one_start_pg_and_show_datadir_env.py

Start PostgreSQL Windows service and print the data directory.

Order of operations:
  1) Load .env (default: ./.env) to populate PGHOST, PGPORT, PGUSER, PGPASSWORD, etc.
  2) Ensure the Windows service is RUNNING (via `sc start`), with optional auto-elevation
  3) Wait for TCP port to accept connections
  4) Prefer: `psql -Atc "SHOW data_directory;"` using env creds (PG* vars)
  5) Fallback: Registry (EDB installer) or parse service command line (-D "<path>")

Works from both Command Prompt and PowerShell.
"""

import argparse
import ctypes
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from typing import Optional, Tuple

# --- .env support ---
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:  # allow script to run even if dotenv not installed
    def load_dotenv(*args, **kwargs):
        pass

try:
    import winreg
except ImportError:
    winreg = None  # non-Windows safeguard


# ------------------------- Defaults (env-aware) -------------------------
ENV_DEFAULTS = {
    "PGHOST": os.environ.get("PGHOST", "127.0.0.1"),
    "PGPORT": os.environ.get("PGPORT", "5432"),
    "PGUSER": os.environ.get("PGUSER", "postgres"),
    "PGPASSWORD": os.environ.get("PGPASSWORD"),  # may be None
}
DEFAULT_SERVICE = os.environ.get("PGSERVICE_WIN", "postgresql-x64-17")
START_TIMEOUT_S = int(os.environ.get("PG_START_TIMEOUT_S", "60"))
LISTEN_TIMEOUT_S = int(os.environ.get("PG_LISTEN_TIMEOUT_S", "30"))
PSQL_TIMEOUT_S = int(os.environ.get("PG_PSQL_TIMEOUT_S", "12"))
# -----------------------------------------------------------------------

log = logging.getLogger("one_pgstarter")


def is_user_admin() -> bool:
    """Return True if current process has admin rights (Windows)."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_as_admin():
    """Re-launch this script with elevation."""
    params = " ".join(shlex.quote(a) for a in sys.argv)
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
    if rc <= 32:
        log.error("Elevation request denied or failed (code=%s).", rc)
        sys.exit(5)
    sys.exit(0)


def run(cmd: list[str] | str) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess (text mode)."""
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    return subprocess.run(cmd, text=True, capture_output=True, shell=shell)


# --------------------- Service helpers (sc) ---------------------
def sc_query(service: str) -> str:
    return run(["sc", "query", service]).stdout


def sc_qc(service: str) -> str:
    return run(["sc", "qc", service]).stdout


def get_service_state(service: str) -> Optional[str]:
    """
    Parse `STATE` line from `sc query`:
      STATE              : 4  RUNNING
    Returns RUNNING | STOPPED | START_PENDING | STOP_PENDING | etc.
    """
    out = sc_query(service)
    for line in out.splitlines():
        if "STATE" in line.upper():
            parts = line.strip().split()
            if parts:
                return parts[-1].strip().upper()
    return None


def ensure_service_started(service: str, elevate: bool = True) -> bool:
    """Start service with `sc start`, with optional auto-elevation."""
    state = get_service_state(service)
    if state == "RUNNING":
        log.info("Service %s already RUNNING.", service)
        return True

    if not is_user_admin():
        if elevate:
            log.warning("Not running as Administrator; relaunching elevated…")
            relaunch_as_admin()
        else:
            log.error("Not admin; cannot start service %s.", service)
            return False

    log.info("Starting service via: sc start %s", service)
    res = run(["sc", "start", service])
    if res.returncode != 0:
        log.error("`sc start` failed (code=%s). stdout:\n%s\nstderr:\n%s",
                  res.returncode, res.stdout.strip(), res.stderr.strip())
        # Common fix: enable autostart, then retry
        log.info("Attempting to set StartupType=auto and retry…")
        run(["sc", "config", service, "start=", "auto"])
        res2 = run(["sc", "start", service])
        if res2.returncode != 0:
            log.error("Retry `sc start` failed. stdout:\n%s\nstderr:\n%s",
                      res2.stdout.strip(), res2.stderr.strip())
            return False
    return True


def wait_for_service_running(service: str, timeout_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = get_service_state(service)
        if st == "RUNNING":
            log.info("Service is RUNNING.")
            return True
        time.sleep(1.0)
    log.error("Timed out waiting for service to RUNNING. Last state: %s", get_service_state(service))
    return False


# --------------------- Network/psql helpers ---------------------
def wait_for_port(host: str, port: int, timeout_s: int) -> bool:
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            s.close()
            log.info("Port %s is accepting connections.", port)
            return True
        except Exception as e:
            last_err = e
            time.sleep(1.0)
        finally:
            try:
                s.close()
            except Exception:
                pass
    log.error("Timed out waiting for %s:%s. Last error: %s", host, port, last_err)
    return False


def try_psql_show_datadir(psql_path: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """
    Run `psql -Atc "SHOW data_directory;"` using PG* environment variables from os.environ.
    Returns (ok, datadir, note).
    """
    binname = psql_path or "psql"
    host = os.environ.get("PGHOST", ENV_DEFAULTS["PGHOST"])
    port = os.environ.get("PGPORT", ENV_DEFAULTS["PGPORT"])
    user = os.environ.get("PGUSER", ENV_DEFAULTS["PGUSER"])

    cmd = [binname, "-h", host, "-p", str(port), "-U", user, "-At", "-c", "SHOW data_directory;"]
    log.info("Attempting via psql: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, text=True, capture_output=True, timeout=PSQL_TIMEOUT_S)
    except FileNotFoundError:
        return (False, None, f"{binname} not found")
    except subprocess.TimeoutExpired:
        return (False, None, "psql timed out (likely waiting for password)")

    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if res.returncode == 0 and out:
        return (True, out, err)
    return (False, None, f"psql failed (code={res.returncode}). stderr: {err or '<empty>'}")


# --------------------- Registry / service -D ---------------------
def reg_read_pg_installs_for_service(service: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (Data Directory, Base Directory) for given Service ID from Registry.
    Works for EDB Windows installer.
    """
    if not winreg:
        return (None, None)

    def scan(root_path: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_path) as base:
                i = 0
                while True:
                    try:
                        subname = winreg.EnumKey(base, i)
                        i += 1
                    except OSError:
                        break
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_path + "\\" + subname) as sub:
                        try:
                            svc_id, _ = winreg.QueryValueEx(sub, "Service ID")
                            if str(svc_id).strip().lower() == service.lower():
                                datadir, _ = winreg.QueryValueEx(sub, "Data Directory")
                                basedir, _ = winreg.QueryValueEx(sub, "Base Directory")
                                return (datadir, basedir)
                        except FileNotFoundError:
                            continue
        except FileNotFoundError:
            pass
        return (None, None)

    # 64-bit
    dd, bd = scan(r"SOFTWARE\PostgreSQL\Installations")
    if dd or bd:
        return (dd, bd)
    # WOW6432Node (rare for PG)
    return scan(r"SOFTWARE\WOW6432Node\PostgreSQL\Installations")


def parse_dashD_from_sc_qc(service: str) -> Optional[str]:
    """
    Parse `BINARY_PATH_NAME` in `sc qc` output to extract -D "<path>".
    """
    qc = sc_qc(service)
    text = " ".join(line.strip() for line in qc.splitlines())
    m = re.search(r'-D\s+"([^"]+)"', text)
    if m:
        return m.group(1)
    m = re.search(r'-D\s+([^\s]+)', text)
    if m:
        return m.group(1)
    return None


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Start PostgreSQL service and print data directory.")
    ap.add_argument("--service", default=DEFAULT_SERVICE, help="Windows service name (default: %(default)s)")
    ap.add_argument("--env", default=".env", help="Path to a .env file to load (default: %(default)s). Set to 'none' to skip.")
    ap.add_argument("--no-elevate", action="store_true", help="Do not auto-elevate; fail if not admin")
    ap.add_argument("--quiet", action="store_true", help="Only print the directory on success")
    args = ap.parse_args()

    # logging
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    log.setLevel(logging.INFO if not args.quiet else logging.WARNING)
    log.addHandler(h)

    # 0) Load .env (if requested)
    if args.env.lower() != "none":
        if os.path.exists(args.env):
            load_dotenv(args.env)
            log.info("Loaded environment from %s", args.env)
        else:
            # Also try default .env in current dir if --env points elsewhere
            if args.env != ".env" and os.path.exists(".env"):
                load_dotenv(".env")
                log.info("Loaded environment from ./.env (fallback)")
            else:
                log.info("No .env file found to load (looked for: %s). Continuing.", args.env)

    # 1) Ensure service started
    log.info("Service: %s  Host: %s  Port: %s  User: %s",
             args.service,
             os.environ.get("PGHOST", ENV_DEFAULTS["PGHOST"]),
             os.environ.get("PGPORT", ENV_DEFAULTS["PGPORT"]),
             os.environ.get("PGUSER", ENV_DEFAULTS["PGUSER"]))

    if not ensure_service_started(args.service, elevate=not args.no_elevate):
        # Try to print datadir anyway
        dd, _ = reg_read_pg_installs_for_service(args.service)
        if not dd:
            dd = parse_dashD_from_sc_qc(args.service)
        if dd:
            print(dd)
            return 0
        sys.exit(2)

    if not wait_for_service_running(args.service, START_TIMEOUT_S):
        dd, _ = reg_read_pg_installs_for_service(args.service)
        if not dd:
            dd = parse_dashD_from_sc_qc(args.service)
        if dd:
            print(dd)
            return 0
        sys.exit(3)

    # 2) Wait for TCP port
    host = os.environ.get("PGHOST", ENV_DEFAULTS["PGHOST"])
    port = int(os.environ.get("PGPORT", ENV_DEFAULTS["PGPORT"]))
    wait_for_port(host, port, LISTEN_TIMEOUT_S)

    # 3) Try psql first (PATH)
    ok, dd, note = try_psql_show_datadir()
    if not ok:
        # Try psql from Base Directory if we know it
        _, basedir = reg_read_pg_installs_for_service(args.service)
        if basedir:
            candidate = os.path.join(basedir, "bin", "psql.exe")
            if os.path.exists(candidate):
                ok, dd, note = try_psql_show_datadir(psql_path=candidate)

    if ok and dd:
        print(dd)
        return 0

    if "timed out" in note.lower() and not os.environ.get("PGPASSWORD"):
        log.info("psql likely waited for a password. Provide PGPASSWORD in your .env to avoid timeouts.")

    log.info("Using fallback because psql didn't return a directory (%s).", note)

    # 4) Registry fallback
    dd, _ = reg_read_pg_installs_for_service(args.service)
    if dd:
        print(dd)
        return 0

    # 5) Parse -D from service command line
    dd = parse_dashD_from_sc_qc(args.service)
    if dd:
        print(dd)
        return 0

    log.error("Unable to determine data directory.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
