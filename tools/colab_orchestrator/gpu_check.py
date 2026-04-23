# Executed via `colab exec -f`. Emits GPU_NAME=... only if code actually ran on
# the VM — so a missing marker means the exec path failed, not the hardware.
import os, subprocess, torch
ok = torch.cuda.is_available()
print(f"GPU_NAME={torch.cuda.get_device_name(0) if ok else 'NONE'}")
print(f"CUDA_AVAILABLE={ok} torch={torch.__version__} cuda={torch.version.cuda}")
print(f"VCPUS={os.cpu_count()}")
print("RAM  ", subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1])
print("DISK ", subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[1])
