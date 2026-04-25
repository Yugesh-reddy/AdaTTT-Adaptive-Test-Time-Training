#!/usr/bin/env bash
# AdaTTT Phase 2 — VM setup. Val2014 + CLIP only. Session C after ready.
# Runs ON the Colab VM. Idempotent on image count. Does not fetch train2014.
set -euo pipefail
ts() { date -u +%H:%M:%S; }
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
echo "[$(ts)] setup start"

# 1. Preflight: accelerator + disk. Fail loudly on anything but an A100.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /content | tail -1
python - <<'PY'
import json, os, subprocess, torch
assert torch.cuda.is_available(), "CUDA not visible"
name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | {name}")
assert "A100" in name, f"expected an A100, got {name}"
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
df = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.strip()
payload = {
    "nvidia_smi": smi,
    "df_content": df,
    "gpu_name": name,
    "cuda_available": True,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
os.makedirs("/content", exist_ok=True)
open("/content/preflight.json", "w").write(json.dumps(payload, indent=2))
print("PREFLIGHT_OK")
print(smi)
print(df)
PY

# 2. Code + VQA val annotations from the uploaded tarball.
tar --warning=no-unknown-keyword -xzf /content/adattt_phase2.tgz -C /content
cd /content/AdaTTT/data
for z in v2_Annotations_Val_mscoco v2_Questions_Val_mscoco; do
  unzip -oq "$z.zip"
done

# 3. COCO val2014 only (~6 GB zip / ~13 GB unzipped). Skip train2014.
count() { if [ -d "$1" ]; then ls "$1" | wc -l; else echo 0; fi; }
LOWIO=""; command -v ionice >/dev/null 2>&1 && LOWIO="ionice -c 3"
fetch() {
  local split=$1 n=$2 dir="/content/AdaTTT/data/$1" have
  have=$(count "$dir")
  if [ "$have" -eq "$n" ]; then echo "[$(ts)] $split already complete"; return 0; fi
  echo "[$(ts)] $split: downloading"
  wget -c -nv --timeout=60 --tries=20 --waitretry=5 \
    "http://images.cocodataset.org/zips/$split.zip" -O "/content/$split.zip"
  echo "[$(ts)] $split: unzipping"
  $LOWIO nice -n 19 unzip -q -o "/content/$split.zip" -d /content/AdaTTT/data/
  rm -f "/content/$split.zip"
  have=$(count "$dir")
  echo "[$(ts)] $split: $have images (expected $n)"
  [ "$have" -eq "$n" ] || { echo "[$(ts)] IMAGE COUNT MISMATCH: $split"; exit 1; }
}
fetch val2014 40504

# 4. CLIP encoder weights only (no ViT/BERT; CLIP-LN is a priced skip).
python - <<'PY'
from transformers import CLIPTextModel, CLIPTokenizerFast, CLIPVisionModel
CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
print("CLIP encoder weights cached")
PY
python -c "import yaml, scipy, tqdm, torchvision; print('imports ok')"
df -h /content | tail -1
echo "[$(ts)] SETUP_DONE"

# 5. Auto-launch once the Mac has uploaded the CLIP checkpoint and armed ready.
while [ ! -f /content/launch.ready ]; do sleep 10; done
args=$(cat /content/resume.args 2>/dev/null || true)
echo "[$(ts)] auto-launch ${args:+($args)}"
bash /content/vm_launch.sh $args
