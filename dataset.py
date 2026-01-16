import pandas as pd
import glob

files = glob.glob("archive/*.csv")
combined_df = pd.concat(
    (pd.read_csv(f) for f in files),
    ignore_index=True,
    sort=False
)

combined_df.to_csv("cicids_2017_combined.csv", index=False)
