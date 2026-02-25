import os
import subprocess
import sys
import platform

"""
It's extremely important to prevent the stacking of multiple processes on the same port, this may cause error and mixed output if the rollout is changed.
"""

def kill_process_on_port(port):
    """Kill the process listening on the specified port."""
    system = platform.system()

    try:
        port = int(port)
    except (ValueError, TypeError):
        print(f"[WARNING] Invalid port number: {port}")
        return

    try:
        if system == "Windows":
            result = subprocess.check_output(
                f'netstat -ano | findstr :{port}',
                shell=True, stderr=subprocess.STDOUT
            ).decode()
            for line in result.splitlines():
                if f":{port}" in line:
                    parts = line.strip().split()
                    pid = parts[-1]
                    if pid.isdigit():
                        subprocess.check_call(f'taskkill /F /PID {pid}', shell=True)
                        print(f"[INFO] Killed process PID={pid} on port {port}")
                        return
        else:
            try:
                result = subprocess.check_output(
                    ["lsof", "-i", f":{port}"], 
                    stderr=subprocess.DEVNULL
                ).decode()
                pids = set()
                for line in result.splitlines()[1:]:
                    parts = line.strip().split()
                    if parts:
                        pids.add(parts[1])

                for pid in pids:
                    os.kill(int(pid), 9)
                    print(f"[INFO] Killed process PID={pid} on port {port}")

            except subprocess.CalledProcessError:
                try:
                    subprocess.check_call(["fuser", "-k", f"{port}/tcp"])
                    print(f"[INFO] Killed process on port {port} using fuser")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print(f"[WARNING] Could not find any process or tool to kill process on port {port}")
    except Exception as e:
        print(f"[ERROR] Failed to kill process on port {port}: {e}")