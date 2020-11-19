#%% Read csv
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
from sqlalchemy.dialects import postgresql
from ast import literal_eval

df = pd.read_csv("../assets/author_dataset_1000_uum.csv")
cols = [i for i in range(1,301)]

# df['vector'] = 




# %%
