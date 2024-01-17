# Imports
import pandas as pd
import tfidf_generation
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from text_tokenization import tokenizeText
from cosine_similarity import compute_cosine_similarities

### DATA EXTRACTION ###

# Getting RetA, RetB combined dataframe
combined_data_clusters = pd.read_csv('../data/processed/combined_data_tokens.csv') # Combined data with titles and tokens


### TF-IDF MATRIX ###

# Generating TF-IDF matrices
combined_tfidf_mtx = tfidf_generation.generate_tfidf_matrix(combined_data_clusters, combined_data_clusters, tokenizeText, lemmas=False)

# Split the TF-IDF representation back into retA and retB
split_index = combined_data_clusters.shape[0]
tfidf_mtx_retA = combined_tfidf_mtx[:split_index]
tfidf_mtx_retB = combined_tfidf_mtx[split_index:]

### COSINE SIMILARITY MATRIX ###

# Compute cosine similarity matrix
cosine_sim_mtx = compute_cosine_similarities(tfidf_mtx_retA, tfidf_mtx_retB)
# Distance matrix
distance_mtx = 1 - cosine_sim_mtx

### DIMENTIONALITY REDUCTION ###

# Dimentionality reduction with PCA
pca_reducer = PCA(n_components=2)
reduced_features = pca_reducer.fit_transform(distance_mtx)

### CLUSTERING ###

# Clustering using HDBSCAN
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=7, min_samples=2, gen_min_span_tree=True)
cosine_sim_df = pd.DataFrame(cosine_sim_mtx, index=combined_data_clusters['title'], columns=combined_data_clusters['title']) # Intermediate dataframe
cosine_sim_df['cluster'] = hdbscan_clusterer.fit_predict(reduced_features)

# Clustering analysis
n_clusters = len(cosine_sim_df['cluster'].unique())

# Saving for analysis
cosine_sim_df.reset_index()[['title', 'cluster']].to_csv('../data/entity-clustering/RetA_RetB_clusters.csv')