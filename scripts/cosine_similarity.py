# Imports
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarities(matrix_a, matrix_b):
    """
    Compute cosine similarities between two matrices.

    Parameters:
        matrix_a (scipy.sparse.csr_matrix): The first matrix for which cosine similarities will be calculated.
        matrix_b (scipy.sparse.csr_matrix): The second matrix for which cosine similarities will be calculated.

    Returns:
        cosine_similarities (numpy.ndarray): A 2D array of cosine similarity scores between the rows of matrix_a and matrix_b.
            Each element (i, j) in the array represents the cosine similarity score between the i-th row of matrix_a and the j-th row
            of matrix_b.

    Example usage:
    >>> matrix_a = generate_tfidf_matrix(data_A, data_B, custom_tokenizer)
    >>> matrix_b = generate_tfidf_matrix(data_C, data_D, custom_tokenizer)
    >>> similarities = compute_cosine_similarities(matrix_a, matrix_b)
    """
    # Calculate cosine similarities between matrix_a and matrix_b and round the results to 5 decimal places
    cosine_similarities = cosine_similarity(matrix_a, matrix_b).round(5)

    return cosine_similarities

