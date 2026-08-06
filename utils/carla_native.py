"""
Native CARLA lifecycle helpers, shared by bo_optimizer.py and rraaa.py.

Launches CarlaUE4.sh directly on the host (bypassing the docker carla_ue4/
carla_ue5 services, which hit a GPU-passthrough rendering problem —
`libGL error: failed to load driver: nouveau`) and manages teardown,
including detached UE4Editor engine processes that escape the launcher's
process group.
"""

import os
import time
import signal
import socket
import subprocess

CARLA_LAUNCH_SCRIPT = os.path.expanduser('~/works/carla/CarlaUE4.sh')
CARLA_HOST          = 'localhost'
CARLA_PORT          = 2000


def _carla_port_open(timeout=2.0):
    try:
        with socket.create_connection((CARLA_HOST, CARLA_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def start_carla(ready_timeout=180):
    """Launch CarlaUE4.sh natively and block until its RPC port is reachable."""
    print("  Starting CARLA (native UE4)...")
    proc = subprocess.Popen(
        [CARLA_LAUNCH_SCRIPT],
        cwd=os.path.dirname(CARLA_LAUNCH_SCRIPT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,   # own process group → can be killed cleanly as a unit
    )
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("CARLA process exited before becoming ready.")
        if _carla_port_open():
            print(f"  CARLA is up and listening on {CARLA_HOST}:{CARLA_PORT}.")
            return proc
        time.sleep(2)
    stop_carla(proc)
    raise RuntimeError(f"CARLA did not open port {CARLA_PORT} within {ready_timeout}s.")


def _carla_engine_pids():
    """PIDs of any UE4Editor process running *this* CARLA project.

    CarlaUE4.sh runs UE4Editor as a plain foreground child (no `exec`), and
    UE4's -game runtime commonly calls setsid() internally — detaching from
    the launcher's process group so os.killpg(launcher_pgid, ...) misses it.
    That left orphaned UE4Editor processes holding port 2000 and GPU memory,
    which then crashed the *next* CARLA instance with a port-bind SIGSEGV.
    Matching on the project path scopes this to our CARLA, not other users'.
    """
    carla_root = os.path.dirname(CARLA_LAUNCH_SCRIPT)
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"UE4Editor.*{carla_root}/Unreal/CarlaUE4/CarlaUE4.uproject"],
            text=True,
        )
        return [int(p) for p in out.split()]
    except subprocess.CalledProcessError:
        return []


def stop_carla(proc, timeout=30):
    """Terminate CARLA (launcher + detached engine) and confirm the port is freed."""
    print("  Stopping CARLA (releasing GPU memory)...")

    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=timeout)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

    # Reap any UE4Editor engine process that detached from the launcher's
    # process group and survived the killpg above.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _carla_engine_pids()
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + timeout
        while time.time() < deadline and _carla_engine_pids():
            time.sleep(1)

    # Confirm port 2000 is actually released before handing back control —
    # otherwise the next start_carla() collides and crashes (SIGSEGV in
    # FCarlaServer::Start while binding the RPC socket).
    deadline = time.time() + timeout
    while time.time() < deadline and _carla_port_open(timeout=1.0):
        time.sleep(1)
    if _carla_port_open(timeout=1.0):
        print(f"  WARNING: port {CARLA_PORT} still open after stop_carla — next launch may collide.")

    print("  CARLA stopped.")
