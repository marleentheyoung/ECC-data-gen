import os
import pandas as pd
import pyreadr

path = '/Users/marleendejonge/Desktop/ECC-data-generation/data/asset_pricing/UMC_monthly_AR.rda'
assert os.path.exists(path), f"File not found: {path}"

res = pyreadr.read_r(path)  # works for .rda and .rds
print("Objects found in file:", list(res.keys()))

for name, obj in res.items():
    outname = (name or "object")  # .rds may have None as key
    df = None

    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, pd.Series):
        # turn a vector/series into a 2-col dataframe and keep index if present
        df = obj.to_frame(name=obj.name or "value").reset_index(drop=False)
    else:
        # try common conversions (matrices, lists of equal-length vectors, etc.)
        try:
            df = pd.DataFrame(obj)
            # if it became a single column of dicts, try to normalize
            if df.shape[1] == 1 and df.iloc[0,0] is not None and isinstance(df.iloc[0,0], dict):
                df = pd.json_normalize(df.iloc[:,0])
        except Exception as e:
            print(f"Skipping '{outname}': cannot convert type {type(obj)} → {e}")
            continue

    df.to_csv(f"{outname}.csv", index=False)
    print(f"Wrote {outname}.csv  shape={df.shape}")
