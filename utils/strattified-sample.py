import pandas as pd

df = pd.read_csv("ita_scan.csv")

targets = {
    "I":   200,
    "II":  200,
    "III": 200,
    "IV":  200,
    "V":   200,
    "VI":  200,
}

sampled = []
for fitz, n in targets.items():
    pool = df[df["fitzpatrick"] == fitz]
    k    = min(n, len(pool))
    sampled.append(pool.sample(k, random_state=42))
    print(f"  Fitzpatrick {fitz}: {k} (pool size: {len(pool)})")

selected = pd.concat(sampled).reset_index(drop=True)
selected.to_csv("selected_identities_stratified.csv", index=False)
print(f"\nTotal selected: {len(selected)}")