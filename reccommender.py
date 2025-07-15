"""
This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""

from pathlib import Path
from typing import Tuple, List
import logging

import implicit
import scipy.sparse as sp

from musiccollaborativefiltering.data import load_user_artists, ArtistRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImplicitRecommender:
    """
    The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model (e.g., ALS, BPR)
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model
        self.user_artists_matrix: sp.csr_matrix = None

    def fit(self, user_artists_matrix: sp.csr_matrix) -> None:
        """
        Fit the implicit model to the user-artists sparse matrix.
        """
        logger.info("Fitting model to user-artist data...")
        self.user_artists_matrix = user_artists_matrix
        self.implicit_model.fit(user_artists_matrix)
        logger.info("Model fitting completed.")

    def recommend(
        self,
        user_id: int,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """
        Return the top `n` recommendations for the given `user_id`.

        Returns:
            - List of recommended artist names
            - List of associated confidence scores
        """
        if self.user_artists_matrix is None:
            raise ValueError("Model has not been fitted yet.")

        if user_id >= self.user_artists_matrix.shape[0]:
            raise IndexError(f"user_id {user_id} is out of bounds.")

        logger.info(f"Generating top {n} recommendations for user {user_id}...")

        try:
            artist_ids, scores = self.implicit_model.recommend(
                user_id, self.user_artists_matrix[user_id], N=n
            )
            artists = [
                self.artist_retriever.get_artist_name_from_id(artist_id)
                for artist_id in artist_ids
            ]
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [], []

        logger.info("Recommendation generation completed.")
        return artists, scores

    def recommend_pretty_print(self, user_id: int, n: int = 10) -> None:
        """
        Print recommendations in a user-friendly format.
        """
        artists, scores = self.recommend(user_id, n)
        print(f"\nTop {n} recommendations for user {user_id}:\n")
        for i, (artist, score) in enumerate(zip(artists, scores), start=1):
            print(f"{i}. {artist} — Score: {score:.3f}")


def main():
    # Paths
    data_dir = Path("../lastfmdata")
    user_artists_path = data_dir / "user_artists.dat"
    artists_path = data_dir / "artists.dat"

    # Load data
    logger.info("Loading user-artist interaction data...")
    user_artists = load_user_artists(user_artists_path)

    logger.info("Loading artist metadata...")
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(artists_path)

    # Create ALS model
    als_model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=15,
        random_state=42,
    )

    # Train and recommend
    recommender = ImplicitRecommender(artist_retriever, als_model)
    recommender.fit(user_artists)
    recommender.recommend_pretty_print(user_id=2, n=5)


if __name__ == "__main__":
    main()
