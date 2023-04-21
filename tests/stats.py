import numpy as np
from scipy.spatial import distance

from hyperspec.stats import _cosine_similarity


def test_cosine_similarity():
    # Define two example vectors
    vector1 = np.array([1, 2, 3], float)
    vector2 = np.array([4, 5, 6], float)

    # Calculate the expected cosine similarity using the scipy distance function
    expected_similarity = 1.0 - distance.cosine(vector1, vector2)

    # Call the function that calculates the cosine similarity
    actual_similarity = _cosine_similarity(vector1, vector2)

    # Check that the actual similarity matches the expected similarity
    assert actual_similarity == expected_similarity
