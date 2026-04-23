# Executed in the Colab kernel via `colab exec -f`. One status line per run,
# plus GPU and host memory — the concurrent runs share both.
import os, subprocess

for name in ("vitbert", "clip"):
    path = f"/content/AdaTTT/logs/train_{name}.log"
    text = open(path).read() if os.path.exists(path) else ""
    lines = text.splitlines()
    epochs = [l for l in lines if "| Val accuracy" in l]
    batches = [l for l in lines if ", Batch " in l]
    pid_file = f"/content/{name}.pid"
    alive = os.path.exists(pid_file) and os.path.exists(f"/proc/{open(pid_file).read().strip()}")
    print(
        f"{name:8s} alive={alive} epochs={len(epochs)} "
        f"last={epochs[-1].split('Val accuracy:')[1].strip() if epochs else '-'} "
        f"at={batches[-1].split('Batch ')[1].split(',')[0] if batches else '-'} "
        f"done={'Training complete' in text} crash={'Traceback' in text}"
    )
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
mem = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()
print(f"gpu      {gpu}")
print(f"host_ram used={mem[2]}G total={mem[1]}G available={mem[-1]}G")

status = open("/content/sync.status").read().strip() if os.path.exists("/content/sync.status") else "mirror not started"
print(f"drive    mounted={os.path.isdir('/content/drive/MyDrive')} | {status}")
