import pickle 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

ohe = model_data["ohe"]
tfidf = model_data["tfidf"]
features = model_data["features"]
df = model_data["dataframe"]

liked_dishes = set()
disliked_dishes = set()

def recommend(cuisine, spice_level, dietary_type, top = 5, avoid_seafood = False, disliked_ingredient = None):
    copied_df = df.copy()

    if avoid_seafood:
        copied_df = copied_df[copied_df["contain_seafood"] == "no"]

    if disliked_ingredient:
        disliked_list = [i.strip().lower() for i in disliked_ingredient.split(",")]

        for ingredients in disliked_list:
            copied_df = copied_df[~copied_df["main_ingredients"].str.contains(ingredients, na = False)]

    copied_df = copied_df[~copied_df["dish_name"].isin(disliked_dishes)]

    filtered_index = copied_df.index

    
    user_preference = ohe.transform([[cuisine.lower(), dietary_type.lower()]])
    user_spice = np.array([[spice_level]])
    user_ingredients = tfidf.transform([""]).toarray()

    user_vector = np.hstack([user_preference, user_spice, user_ingredients])

    similarity = cosine_similarity(user_vector, features)

    similarity_scores = similarity[0][filtered_index]

    for i, idx in enumerate(filtered_index):
        if df.loc[idx, "dish_name"] in liked_dishes:
            similarity_scores[i] += 0.1

    top_relative_indices = similarity_scores.argsort()[-top:][::-1]
    top_indices = filtered_index[top_relative_indices]

    return df.loc[top_indices][["dish_name", "cuisine", "spice_level", "dietary_type", "description"]]

def like_dish(dish_name):
    liked_dishes.add(dish_name.lower())
    if dish_name.lower() in disliked_dishes:
        disliked_dishes.remove(dish_name.lower())

def dislike_dish(dish_name):
    disliked_dishes.add(dish_name.lower())
    if dish_name.lower() in liked_dishes:
        liked_dishes.remove(dish_name.lower())