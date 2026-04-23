"""Rebuild the VM payload (adattt_phase1.tgz) from the project and diff it.

Usage: python3 build_payload.py PREVIOUS.tgz

The previous payload's file list is the manifest, so nothing is guessed. New
.py files under ttt/, gpu/ and scripts/ are added and reported. Built with
tarfile, so no macOS xattrs or AppleDouble files reach the VM. Prints every
file whose content differs from the previous payload.
"""
import glob
import hashlib
import io
import os
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get(
    "ORCH_PROJECT",
    "/Users/yugesh/Library/CloudStorage/GoogleDrive-yugeshreddysappidi@gmail.com/My Drive/AdaTTT")
OUT = os.path.join(HERE, "adattt_phase1.tgz")
CFG_VITBERT = os.path.join(PROJECT, "tools", "colab_orchestrator", "cfg_vitbert.yaml")
ROOT = "AdaTTT/"


def sha(data):
    return hashlib.sha256(data).hexdigest()[:12]


def source(rel):
    return CFG_VITBERT if rel == "cfg_vitbert.yaml" else os.path.join(PROJECT, rel)


prev_path = sys.argv[1]
with tarfile.open(prev_path) as prev:
    old = {m.name[len(ROOT):]: sha(prev.extractfile(m).read())
           for m in prev.getmembers() if m.isfile()}

manifest = set(old)
for pattern in ("ttt/*.py", "gpu/*.py", "scripts/*.py"):
    for path in glob.glob(os.path.join(PROJECT, pattern)):
        manifest.add(os.path.relpath(path, PROJECT))
missing = [rel for rel in manifest if not os.path.isfile(source(rel))]
if missing:
    sys.exit(f"manifest files missing from the project: {missing}")

tmp = OUT + ".tmp"
now = time.time()
new = {}
with tarfile.open(tmp, "w:gz") as tar:
    for rel in sorted(manifest):
        data = open(source(rel), "rb").read()
        new[rel] = sha(data)
        info = tarfile.TarInfo(ROOT + rel)
        info.size, info.mtime, info.mode = len(data), now, 0o644
        tar.addfile(info, io.BytesIO(data))
os.replace(tmp, OUT)

added = sorted(set(new) - set(old))
changed = sorted(r for r in set(new) & set(old) if new[r] != old[r])
print(f"wrote {OUT}: {len(new)} files, {os.path.getsize(OUT)/1e6:.1f} MB")
print(f"  added   ({len(added)}): {added}")
print(f"  changed ({len(changed)}): {changed}")
print(f"  unchanged: {len(new) - len(added) - len(changed)}")
