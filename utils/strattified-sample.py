import pandas as pd

df = pd.read_csv("ita_scan.csv")

targets = {
    "I":   167,
    "II":  167,
    "III": 167,
    "IV":  167,
    "V":   166,
    "VI":  166,
}

sampled = []
for fitz, n in targets.items():
    pool = df[df["fitzpatrick"] == fitz]
    k    = min(n, len(pool))
    sampled.append(pool.sample(k, random_state=42))
    print(f"  Fitzpatrick {fitz}: {k}")

selected = pd.concat(sampled).reset_index(drop=True)
selected.to_csv("selected_identities.csv", index=False)
print(f"\nTotal selected: {len(selected)}")