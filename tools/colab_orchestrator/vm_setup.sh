#!/usr/bin/env bash
# AdaTTT Phase 1 — VM setup. Runs ON the Colab VM in the background, stdout
# redirected to /content/setup.log by the bootstrap. Idempotent: a re-run on
# the same VM skips completed image splits.
set -euo pipefail
ts() { date -u +%H:%M:%S; }
# Bounded network waits: a stalled connection must retry, not hang the setup.
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
echo "[$(ts)] setup start"

# 1. The accelerator the user asked for — fail loudly on anything else.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible"
name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | {name}")
assert "A100" in name, f"expected an A100, got {name}"
PY

# 2. Code + VQA annotations from the uploaded tarball.
tar --warning=no-unknown-keyword -xzf /content/adattt_phase1.tgz -C /content
cd /content/AdaTTT/data
for z in v2_Annotations_Train_mscoco v2_Annotations_Val_mscoco \
         v2_Questions_Train_mscoco v2_Questions_Val_mscoco; do
  unzip -oq "$z.zip"
done
cp /content/AdaTTT/cfg_vitbert.yaml /tmp/cfg_vitbert.yaml

# 3. COCO images from the official host — the copies in Drive are truncated.
# count() must not fail on a missing directory: under `set -euo pipefail`, an
# `ls` of a directory that does not exist yet fails the whole $(...) pipeline
# and silently kills the background fetch before wget ever runs.
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
  # Lowest CPU and I/O priority: two concurrent COCO unzips (~123k files) are
  # the heaviest disk load of the whole job, and the first VM's Jupyter
  # websocket stalled while they ran. Keep the server responsive.
  $LOWIO nice -n 19 unzip -q -o "/content/$split.zip" -d /content/AdaTTT/data/
  rm -f "/content/$split.zip"
  have=$(count "$dir")
  echo "[$(ts)] $split: $have images (expected $n)"
  [ "$have" -eq "$n" ] || { echo "[$(ts)] IMAGE COUNT MISMATCH: $split"; exit 1; }
}
fetch train2014 82783 & p1=$!
fetch val2014 40504 & p2=$!
rc=0; wait $p1 || rc=1; wait $p2 || rc=1
[ "$rc" -eq 0 ] || { echo "[$(ts)] COCO FETCH FAILED"; exit 1; }

# 4. Pre-fetch encoder weights so two concurrent runs don't race the HF cache.
python - <<'PY'
from transformers import (BertModel, BertTokenizer, CLIPTextModel,
                          CLIPTokenizerFast, CLIPVisionModel, ViTModel)
ViTModel.from_pretrained("google/vit-base-patch16-224")
BertModel.from_pretrained("bert-base-uncased")
BertTokenizer.from_pretrained("bert-base-uncased")
CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
print("encoder weights cached")
PY
python -c "import yaml, scipy, tqdm, torchvision; print('imports ok')"
df -h /content | tail -1
echo "[$(ts)] SETUP_DONE"

# 5. Auto-launch, so a finished setup never leaves the A100 idle waiting on the
#    orchestrator or the agent. launch.ready is written only after any resume
#    checkpoints are uploaded and verified, so a resume can't start fresh.
while [ ! -f /content/launch.ready ]; do sleep 10; done
args=$(cat /content/resume.args 2>/dev/null || true)
echo "[$(ts)] auto-launch ${args:+($args)}"
bash /content/vm_launch.sh $args
