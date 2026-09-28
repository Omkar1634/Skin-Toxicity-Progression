# batch_run.py
import os
import glob
import argparse
import yaml
import copy
import subprocess
import sys
import csv
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True, help="Base config YAML path")
    p.add_argument("--selected_csv", default=None,  help="Stratified CSV; omit to run ALL images")
    p.add_argument("--start",        type=int, default=0,    help="Start index (inclusive)")
    p.add_argument("--end",          type=int, default=None, help="End index (exclusive)")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    albedo_dir  = base_config["paths"]["ALBEDO_DIR"]
    output_root = base_config["paths"]["OUTPUT_ROOT"]

    # -- build face list -------------------------------------------------------
    if args.selected_csv:
        with open(args.selected_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        albedo_files = [
            os.path.join(albedo_dir, f"{int(r['face_id']):06d}.png")
            for r in rows
        ]
        mode = f"stratified CSV ({args.selected_csv})"
    else:
        albedo_files = sorted(glob.glob(os.path.join(albedo_dir, "*.png")))
        mode = "all images (glob)"

    albedo_files = albedo_files[args.start: args.end]
    print(f"[batch] mode: {mode}")
    print(f"[batch] {len(albedo_files)} faces to process\n")
    # --------------------------------------------------------------------------

    failed  = []
    skipped = 0

    pbar = tqdm(enumerate(albedo_files), total=len(albedo_files),
                unit="face", dynamic_ncols=True)

    for idx, albedo_path in pbar:
        face_id = os.path.splitext(os.path.basename(albedo_path))[0]
        out_dir = os.path.join(output_root, face_id)

        done_marker = os.path.join(out_dir, f"metadata_{face_id}.csv")
        if os.path.exists(done_marker):
            skipped += 1
            pbar.set_postfix(face=face_id, status="skip", failed=len(failed))
            continue

        pbar.set_postfix(face=face_id, status="run ", failed=len(failed))

        cfg = copy.deepcopy(base_config)
        cfg["paths"]["ALBEDO_PATH"] = albedo_path
        cfg["paths"]["OUTPUT_DIR"]  = out_dir
        os.makedirs(out_dir, exist_ok=True)

        tmp_config = os.path.join(out_dir, f"config_{face_id}.yaml")
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        result = subprocess.run(
            [sys.executable, "erythema_sweep.py", "--config", tmp_config],
            stdout=None,          # streams live to terminal
            stderr=subprocess.PIPE  # captured for error reporting only
        )

        if result.returncode != 0:
            failed.append(face_id)
            pbar.set_postfix(face=face_id, status="FAIL", failed=len(failed))
            print(f"\n[batch] ERROR {face_id}:\n{result.stderr.decode()}", flush=True)

    pbar.close()
    print(f"\n[batch] done — processed {len(albedo_files)-skipped-len(failed)} | "
          f"skipped {skipped} | failed {len(failed)}")
    if failed:
        print(f"[batch] failed faces: {failed}")

if __name__ == "__main__":
    main()