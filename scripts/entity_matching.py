# Imports fo entity matching
import pandas as pd
import tfidf_generation
from text_tokenization import tokenizeText
from sklearn.metrics.pairwise import cosine_similarity


### READING THE DATA ###

# Getting RetA, RetB and combined dataframes
retA_data_tokens = pd.read_csv('../data/processed/retailerA_tokens.csv') # Retailer A data with titles and tokens
retB_data_tokens = pd.read_csv('../data/processed/retailerB_tokens.csv') # Retailer B data with titles and tokens
combined_data_tokens = pd.read_csv('../data/processed/combined_data_tokens.csv') # Combined data with titles and tokens


### TFIDF MATRIX ###

# Generating TF-IDF matrices
tfidf_mtx = tfidf_generation.generate_tfidf_matrix(retA_data_tokens, retA_data_tokens, tokenizeText)


### COSINE SIMILARITIES ###

# Compute cosine similarities
split_index = retA_data_tokens.shape[0]
tfidf_matrix_ret_a = tfidf_mtx[:split_index]
tfidf_matrix_ret_b = tfidf_mtx[split_index:]
cosine_similarities_ab = cosine_similarity.compute_cosine_similarities(tfidf_matrix_ret_a, tfidf_matrix_ret_b)


### SAVING SIMILARITIES DATAFRAME ###

# Building a dataframe with cosine similarities in records from RetA and RetB based on a threshold
similarities = []
thr = 0.6

for idx, row in enumerate(cosine_similarities_ab):
    # Loop through each row of the cosine similarities matrix
    for col_idx, similarity in enumerate(row):
        # Loop through each element in the row
        if similarity > thr:  # Threshold adjusted for a 60% of similarity
            similarities.append((retA_data_tokens.iloc[idx]['title'], retB_data_tokens.iloc[col_idx]['title'], similarity))

sim_ab_df = pd.DataFrame(similarities, columns=['Title_retA', 'Title_retB', 'Similarity'])

sim_ab_df['Similarity'] = sim_ab_df['Similarity'].round(5) # Create `similarity` column
sim_ab_df.to_csv('../data/processed/RetA_RetB_similarities.csv') # Save the dataframe for later usage