# Executed in the Colab kernel via `colab exec -f`. Starts vm_setup.sh detached
# (new session) so it survives exec disconnects and kernel restarts; progress
# goes to /content/setup.log and is polled separately.
import subprocess
p = subprocess.Popen(
    ["bash", "/content/vm_setup.sh"],
    stdout=open("/content/setup.log", "a"), stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"setup started, pid {p.pid}")
