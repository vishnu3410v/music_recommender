from pathlib import Path

from musiccollaborativefiltering.data import load_user_artists, ArtistRetriever
from musiccollaborativefiltering.recommender import ImplicitRecommender

import implicit


def main():
    # === Load Data ===
    user_artists_path = Path("lastfmdata/user_artists_sample.dat")
    artists_path = Path("lastfmdata/artists_sample.dat")

    user_artists = load_user_artists(user_artists_path)

    retriever = ArtistRetriever()
    retriever.load_artists(artists_path)

    # === Train ALS Model ===
    model = implicit.als.AlternatingLeastSquares(
        factors=64, regularization=0.05, iterations=15
    )

    recommender = ImplicitRecommender(retriever, model)
    recommender.fit(user_artists)

    # === Recommend for a user ===
    user_id = 2
    recommender.recommend_pretty_print(user_id=user_id, n=5)


if __name__ == "__main__":
    main()
