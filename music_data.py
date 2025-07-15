"""
This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""

from pathlib import Path
from typing import Optional

import logging
import scipy.sparse as sp
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_user_artists(user_artists_file: Path) -> sp.csr_matrix:
    """
    Load the user-artists file and return a user-artist matrix in CSR format.

    Args:
        user_artists_file (Path): Path to the user_artists.dat file.

    Returns:
        csr_matrix: Sparse matrix with users as rows and artists as columns.
    """
    if not user_artists_file.exists():
        raise FileNotFoundError(f"File not found: {user_artists_file}")

    logger.info(f"Loading user-artist interactions from: {user_artists_file}")
    df = pd.read_csv(user_artists_file, sep="\t")

    if not {"userID", "artistID", "weight"}.issubset(df.columns):
        raise ValueError("Expected columns: userID, artistID, weight")

    df.set_index(["userID", "artistID"], inplace=True)

    coo = sp.coo_matrix(
        (
            df.weight.astype(float),
            (
                df.index.get_level_values(0),
                df.index.get_level_values(1),
            ),
        )
    )

    csr = coo.tocsr()
    logger.info(f"Loaded matrix shape: {csr.shape}")
    return csr


class ArtistRetriever:
    """
    The ArtistRetriever class retrieves artist names given an artist ID.
    """

    def __init__(self):
        self._artists_df: Optional[pd.DataFrame] = None

    def load_artists(self, artists_file: Path) -> None:
        """
        Load the artists file into a private DataFrame.

        Args:
            artists_file (Path): Path to the artists.dat file.
        """
        if not artists_file.exists():
            raise FileNotFoundError(f"File not found: {artists_file}")

        logger.info(f"Loading artist data from: {artists_file}")
        df = pd.read_csv(artists_file, sep="\t")

        if not {"id", "name"}.issubset(df.columns):
            raise ValueError("Expected columns: id, name")

        self._artists_df = df.set_index("id")
        logger.info(f"Loaded {len(self._artists_df)} artists.")

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """
        Return the artist name from the given artist ID.

        Args:
            artist_id (int): The ID of the artist.

        Returns:
            str: The name of the artist.

        Raises:
            ValueError: If artist data is not loaded or ID is invalid.
        """
        if self._artists_df is None:
            raise ValueError("Artist data not loaded. Call load_artists() first.")

        if artist_id not in self._artists_df.index:
            logger.warning(f"Artist ID {artist_id} not found.")
            return f"Unknown Artist (ID: {artist_id})"

        return self._artists_df.loc[artist_id, "name"]


if __name__ == "__main__":
    try:
        retriever = ArtistRetriever()
        retriever.load_artists(Path("../lastfmdata/artists.dat"))
        print(retriever.get_artist_name_from_id(1))
        print(retriever.get_artist_name_from_id(999999))  # Test unknown ID
    except Exception as e:
        logger.error(f"Error during test run: {e}")
