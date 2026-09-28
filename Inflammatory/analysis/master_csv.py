import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csvs = glob.glob(r"D:\Github\PhD Code\Erythema-Progression\output\1000-identities\*\metadata_*.csv")
master = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
master.to_csv("master_metadata.csv", index=False)
print(f"{len(master)} rows, {master['face_id'].nunique()} faces") 