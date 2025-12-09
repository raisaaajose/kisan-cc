import numpy as np
import pandas as pd
import yaml
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from scipy import sparse

config_path = "../config/params.yaml"
with open(config_path) as f:
    config_data = yaml.safe_load(f)


df_additional = pd.read_csv("../" + config_data["data"]["additional_data"])
df_train = pd.read_csv("../" + config_data["data"]["train"])
output_dir = "../" + config_data["data"]["processed_graph"]
os.makedirs(output_dir, exist_ok=True)
k = int(config_data["model"]["knn"]["k"])
E = float(config_data["model"]["knn"]["E"])

# to find count of fields
valid_ids = df_train[df_train["Quality"] != 1]["Field_ID"]
df_additional = df_additional[df_additional["Field_ID"].isin(valid_ids)]
df_additional = df_additional.drop_duplicates(subset=["Field_ID"], keep="first")
df_additional = df_additional.reset_index(drop=True)
df_additional_features = df_additional.drop(columns=["Field_ID"])
field_count = len(df_additional_features)

# mutual+weighted knn
model = NearestNeighbors(n_neighbors=k)
imputer = SimpleImputer(strategy="mean")

# print("na: {}".format(df_additional_features.isna().any(axis=1).sum()))

df_additional_features = imputer.fit_transform(df_additional_features)
model.fit(df_additional_features)
distances, indices = model.kneighbors(df_additional_features)


row_ptr = np.repeat(np.arange(len(indices)), k)
col_ptr = indices.flatten()
data_ptr = distances.flatten()
A = sparse.csr_matrix((data_ptr, (row_ptr, col_ptr)), shape=(field_count, field_count))

A_transpose = A.transpose()

mutual_mask = A.astype(bool).multiply(A_transpose.astype(bool))
A_final = mutual_mask.multiply(A)
A_final.data = 1.0 / (A_final.data + E)
A_final.setdiag(1.0 / E)

# print(A_final.getnnz(axis=1))

sparse.save_npz(os.path.join(output_dir, "adj_matrix.npz"), A_final)
np.save(os.path.join(output_dir, "node_features.npz"), df_additional_features)
np.save(os.path.join(output_dir, "graph_field_id.npy"), df_additional["Field_ID"])

print(f"Files saved to {output_dir}")
