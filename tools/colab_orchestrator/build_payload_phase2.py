"""Build the Phase 2 VM payload (adattt_phase2.tgz) from the working tree.

Includes uncommitted shift/loader files from disk. Does not pack feature
caches, Phase 1 checkpoints, or train2014. Built with tarfile so no macOS
xattrs reach the VM.
"""
import glob
import io
import os
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get(
    "ORCH_PROJECT",
    "/Users/yugesh/Library/CloudStorage/GoogleDrive-yugeshreddysappidi@gmail.com/My Drive/AdaTTT",
)
OUT = os.path.join(HERE, "adattt_phase2.tgz")
ROOT = "AdaTTT/"

EXPLICIT = [
    "config/config.yaml",
    "data/eval_subset_8k.json",
    # τ and the Session D step size are fit on this subset on the VM.
    "data/gate_train_subset_8k.json",
    "data/answer_vocab.json",
    "data/v2_Annotations_Val_mscoco.zip",
    "data/v2_Questions_Val_mscoco.zip",
    "tools/colab_orchestrator/vm_setup_phase2.sh",
    "tools/colab_orchestrator/vm_launch_phase2.sh",
]


def files():
    rels = set(EXPLICIT)
    for pattern in ("ttt/*.py", "gpu/*.py"):
        for path in glob.glob(os.path.join(PROJECT, pattern)):
            rels.add(os.path.relpath(path, PROJECT))
    return sorted(rels)


def main():
    rels = files()
    missing = [r for r in rels if not os.path.isfile(os.path.join(PROJECT, r))]
    if missing:
        raise SystemExit(f"payload files missing: {missing}")
    forbidden = [r for r in rels if "features/" in r or r.endswith(".pt")]
    if forbidden:
        raise SystemExit(f"refusing to pack caches/checkpoints: {forbidden}")
    tmp = OUT + ".tmp"
    now = time.time()
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in rels:
            data = open(os.path.join(PROJECT, rel), "rb").read()
            info = tarfile.TarInfo(ROOT + rel)
            info.size, info.mtime, info.mode = len(data), now, 0o644
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(rels)} files, {os.path.getsize(OUT)/1e6:.1f} MB")
    for rel in rels:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
