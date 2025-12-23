import pandas as pd
import numpy as np

# Movie metadata
movies = pd.read_csv(
    "movie_title.csv",
    sep=",",  # adjust if comma-separated
    names=["movieId", "title", "genres"],
    header=0,
    encoding="latin1",
)

movies

ratings_data = []
current_movie = None

with open("combined_data_1.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line.endswith(":"):
            current_movie = int(line.replace(":", ""))
        else:
            userId, rating, _ = line.split(",")  # timestamp ignored
            ratings_data.append([int(userId), current_movie, float(rating)])

ratings = pd.DataFrame(ratings_data, columns=["userId", "movieId", "rating"])

ratings.head()

data = ratings.merge(movies, on="movieId", how="left")

data["genres"] = data["genres"].str.split("|")
data_exploded = data.explode("genres")

data_exploded.head()

# USER INTEREST PROFILING (Genre preference per user)
user_genre_profile = (
    data_exploded.groupby(["userId", "genres"])["rating"].mean().reset_index()
)

genre_popularity = (
    data_exploded.groupby("genres")["rating"].count().sort_values(ascending=False)
)

genre_avg_rating = (
    data_exploded.groupby("genres")["rating"].mean().sort_values(ascending=False)
)

best_genre = genre_avg_rating.idxmax()
worst_genre = genre_avg_rating.idxmin()

from sklearn.preprocessing import LabelEncoder

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["user_enc"] = user_encoder.fit_transform(ratings["userId"])
ratings["movie_enc"] = movie_encoder.fit_transform(ratings["movieId"])

from scipy.sparse import csr_matrix

user_movie_sparse = csr_matrix(
    (ratings["rating"], (ratings["user_enc"], ratings["movie_enc"]))
)

from sklearn.metrics.pairwise import cosine_similarity

item_similarity = cosine_similarity(user_movie_sparse.T, dense_output=False)


def predict_user_scores(user_id):
    user_idx = user_encoder.transform([user_id])[0]
    user_vector = user_movie_sparse[user_idx]

    # Score = sum(similarity × user ratings)
    scores = user_vector.dot(item_similarity)
    scores = scores.toarray().flatten()

    # Remove already watched movies
    scores[user_vector.indices] = -1
    return scores


def recommend_best_per_genre(user_id):
    scores = predict_user_scores(user_id)

    # Attach scores to movie metadata
    score_df = pd.DataFrame(
        {
            "movieId": movie_encoder.inverse_transform(np.arange(len(scores))),
            "score": scores,
        }
    )

    score_df = score_df.merge(movies, on="movieId")
    score_df["genres"] = score_df["genres"].str.split("|")
    score_df = score_df.explode("genres")

    # Pick best movie per genre
    recommendations = (
        score_df.sort_values("score", ascending=False)
        .groupby("genres")
        .head(1)[["genres", "title", "score"]]
    )

    return recommendations


recommend_best_per_genre(user_id=1488844)

genre_popularity = (
    data_exploded.groupby("genres")["rating"].count().sort_values(ascending=False)
)

genre_avg_rating = data_exploded.groupby("genres")["rating"].mean()

best_genre = genre_avg_rating.idxmax()
worst_genre = genre_avg_rating.idxmin()


def personalized_genre_watchlist(user_id):
    recs = recommend_best_per_genre(user_id)
    return recs.sort_values("score", ascending=False)


print("Most Popular Genres:")
print(genre_popularity.head(5))

print("\nBest Rated Genre:", best_genre)
print("Worst Rated Genre:", worst_genre)

test_user = 1488844

print("\nUser Genre Preference Profile:")
print(
    user_genre_profile[user_genre_profile["userId"] == test_user].sort_values(
        "rating", ascending=False
    )
)

scores = predict_user_scores(test_user)

print("\nPredicted Scores Vector:")
print(scores[:10])  # sanity check

genre_recommendations = recommend_best_per_genre(test_user)

print("\nBest Movie Recommendation Per Genre:")
print(genre_recommendations)

final_watchlist = personalized_genre_watchlist(test_user)

print("\nFinal Personalized Watchlist:")
print(final_watchlist)
