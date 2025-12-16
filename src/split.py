import numpy as np
import yaml
import pandas as pd
import os
from sklearn.model_selection import train_test_split

config = yaml.safe_load(open("config/params.yaml"))

train_df = pd.read_csv(config["data"]["train"])
train_df = train_df[train_df["Quality"] != 1]
known_ids = train_df["Field_ID"].values
graph_field_ids = np.load(
    os.path.join(config["data"]["processed"], "graph_field_id.npy"), allow_pickle=True
)
train_ids, val_ids = train_test_split(known_ids, test_size=0.2, random_state=42)

print(f"Total Known Labels: {len(known_ids)}")
print(f"New Training Set: {len(train_ids)}")
print(f"New Validation Set: {len(val_ids)}")

train_mask_split = np.isin(graph_field_ids, train_ids)
val_mask_split = np.isin(graph_field_ids, val_ids)

np.save(
    os.path.join(config["data"]["processed"], "train_mask_split.npy"), train_mask_split
)
np.save(os.path.join(config["data"]["processed"], "val_mask_split.npy"), val_mask_split)
