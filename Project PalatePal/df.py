import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("Dataset.csv")
print(df.head())

df.columns = df.columns.astype(str).str.strip().str.lower()
print(df.head())

df[["dish_name", "main_ingredients", "cuisine", "dietary_type", "contain_seafood"]] = df[["dish_name", "main_ingredients", "cuisine", "dietary_type", "contain_seafood"]].astype(str).apply(lambda x: x.str.lower().str.strip())
print(df.head())

df["dish_name"] = df["dish_name"].str.replace("\n", " ", regex = False)
df["main_ingredients"] = df["main_ingredients"].str.replace("\n", " ", regex = False)
print(df.head())

df["description"] = df["description"].fillna("Delicious Dish").str.lower()
print(df.head())

categorical_columns = df[["cuisine", "dietary_type"]]

one_hot = OneHotEncoder(sparse_output=False)
encoded_columns = one_hot.fit_transform(categorical_columns)

spice = df[["spice_level"]].values

tfidf = TfidfVectorizer()
ingredient_matrix = tfidf.fit_transform(df["main_ingredients"]).toarray()

final_features = np.hstack([encoded_columns, spice, ingredient_matrix])

model_data = {
    "ohe" : one_hot,
    "tfidf" : tfidf,
    "features" : final_features,
    "dataframe" : df
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("The model saved successfully!")