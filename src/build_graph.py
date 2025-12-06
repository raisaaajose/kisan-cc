import numpy as np
import pandas as pd
import yaml

config_path = "../../config/params.yaml"
config_data = yaml.safe_load(config_path)

df = pd.read_csv(config_data['data']['additional_data'])

print(df.head)