# pre_scan_ita.py  — run this first, takes ~10 min for 10k faces
import os, glob, csv, re, sys
import cv2
from skin_tone import face_ita  
from helper import ita_to_fitzpatrick
import yaml 
import argparse
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def update(self, n=1):
            pass

PROJECT_ROOT = sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Config.yaml", help="YAML config path")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
    
    
args = parse_args()
config = load_config(args.config)

out_csv    = "ita_scan.csv"

albedo_dir = config["paths"]["ALBEDO_DIR"]   # adjust to your config key

# -- collect all PNGs, group by 6-digit base ID -------------------------
BASE_RE    = re.compile(r"^(\d{6})$")
VARIANT_RE = re.compile(r"^(\d{6})_cof_a\d+_final_uv$")

candidates = defaultdict(dict)   # {base_id: {"plain": path, "cof": path}}

for p in tqdm(sorted(glob.glob(os.path.join(albedo_dir, "*.png"))), desc="Scanning albedo files"):
    stem = os.path.splitext(os.path.basename(p))[0]
    m = BASE_RE.fullmatch(stem)
    if m:
        candidates[m.group(1)]["plain"] = p
        continue
    m = VARIANT_RE.fullmatch(stem)
    if m:
        candidates[m.group(1)]["cof"] = p

# -- pick best albedo per identity: prefer corrected (_cof_) if present --
albedo_files = []
for base_id in tqdm(sorted(candidates), desc="Selecting albedo files", leave=False):
    slot = candidates[base_id]
    albedo_files.append((base_id, slot.get("cof") or slot.get("plain")))

print(f"[scan] {len(albedo_files)} unique identities found "
      f"({sum(1 for _, p in albedo_files if '_cof_' in p)} corrected, "
      f"{sum(1 for _, p in albedo_files if '_cof_' not in p)} plain)")

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["face_id", "ita", "band", "fitzpatrick"])
    for base_id, p in tqdm(albedo_files, desc="Processing faces", unit="face"):
        try:
            face_id = base_id
            # try standard read first
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                # try EXR
                img = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                print(f"[skip] {face_id}: unreadable file")
                continue
            r = face_ita(p, mask=None)
            w.writerow([face_id, round(r["ita"], 4), r["band"],
                        ita_to_fitzpatrick(r["ita"])])
        except Exception as e:
            print(f"[skip] {face_id}: {e}")

print(f"[scan] done → {out_csv}")