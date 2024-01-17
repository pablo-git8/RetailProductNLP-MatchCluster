# Imports
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tfidf_matrix(data_A, data_B, tokenize_function, lemmas):
    """
    Generate a TF-IDF matrix from the titles of two datasets, data_A and data_B.

    Parameters:
        data_A (pd.DataFrame): A pandas DataFrame containing a 'title' column with text data for the first dataset.
        data_B (pd.DataFrame): A pandas DataFrame containing a 'title' column with text data for the second dataset.
        tokenize_function (callable): A function that tokenizes and preprocesses text data. It should accept a text input and 
            return a list of tokens.

    Returns:
        tfidf_matrix (scipy.sparse.csr_matrix): A sparse TF-IDF matrix representing the combined titles of data_A and data_B. 
            Rows correspond to documents (titles), and columns correspond to unique terms (words) in the corpus.

    Example usage:
    >>> data_A = pd.DataFrame({'title': ['Document 1', 'Document 2']})
    >>> data_B = pd.DataFrame({'title': ['Document 3', 'Document 4']})
    >>> tfidf_matrix = generate_tfidf_matrix(data_A, data_B, custom_tokenizer)
    """
    # Initialize a TF-IDF vectorizer with a custom tokenizer function
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda text: tokenize_function(text, lemmas=lemmas))

    # Fit the vectorizer on the combined titles of data_A and data_B and transform the data into a TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_A['title'].tolist() + data_B['title'].tolist())

    return tfidf_matrix

