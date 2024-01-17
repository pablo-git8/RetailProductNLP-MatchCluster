# RetailProductNLP-MatchCluster

## Project Overview

This project is centered around Natural Language Processing (NLP) for an entity matching and clustering use case. It aims to compare products listed by two different retailers, Retailer A and Retailer B. The project applies various NLP techniques including tokenization, TF-IDF vectorization, clustering algorithms, and dimensionality reduction. The entire workflow is encapsulated in a pipeline for easy implementation and scalability.


### Key Components

- **requirements.txt**: Lists all the Python dependencies required for the project.
- **data/**: Contains various subfolders for raw data, processed data, and output from entity matching and clustering.
- **documentation/**: Includes notebooks for problem statement and images illustrating the problem statement and methodology.
- **notebooks-dev/**: Jupyter notebooks for development and exploratory analysis, including data extraction, processing, and modeling.
- **pipeline/**: Shell scripts (`run_pipeline.sh`) to execute the whole pipeline.
- **scripts/**: Python scripts for each step of the process, including data extraction, text tokenization, TF-IDF generation, cosine similarity computation, entity matching, and clustering.



### Project Workflow

1. **Data Extraction and Preprocessing**: Raw data from Retailer A and B are processed and tokenized.
2. **TF-IDF Vectorization**: Implements TF-IDF to convert text data into numerical form.
3. **Entity Matching**: Uses cosine similarity to find similarities between products from Retailer A and B.
4. **Entity Clustering**: Applies clustering algorithms to group similar entities.
5. **Pipeline Integration**: The entire process is streamlined into a pipeline for reproducibility and scalability.


### Example Outputs

This section provides a glimpse into the kind of outputs generated by the project, showcasing practical examples of the entity matching and clustering processes.

#### Entity Matching
- **Similarity Scores Dataframe**: A dataframe showing the computed similarity scores for products in Retailer A and Retailer B. This table provides a direct comparison between products, highlighting how closely related they are in terms of their descriptions or other text-based features.

<p align="center">
	<img src="https://raw.githubusercontent.com/pablo-git8/RetailProductNLP-MatchCluster/main/documentation/images/entity_matching_output_df.png" alt="400" width="600"/>
</p>

- **Similarity Distribution Histogram**: Histogram representing the distribution of similarity scores. This visual aid helps in understanding the general pattern of similarities between products – for instance, how many pairs have high similarity scores versus low scores, indicating the effectiveness of the matching algorithm.

<p align="center">
	<img src="https://raw.githubusercontent.com/pablo-git8/RetailProductNLP-MatchCluster/main/documentation/images/similarities_distribution.png" alt="300" width="600"/>
</p>

#### Entity Clustering
- **Clustered Titles**: An example output showing how product titles from both Retailer A and Retailer B are concatenated and assigned to clusters. Each cluster represents a group of similar products, providing insights into how products from different retailers can be grouped based on their textual features.

<p align="center">
	<img src="https://raw.githubusercontent.com/pablo-git8/RetailProductNLP-MatchCluster/main/documentation/images/entity_clustering_output_df.png" alt="300" width="600"/>
</p>

These example outputs serve as a reference to understand the practical applications of the project and the insights that can be derived from its analysis.

# Future Work

The project has demonstrated significant capabilities in NLP for entity matching and clustering between different retailers. However, there is always room for improvement and expansion. The following are key areas identified for future development:

1. **Pipeline Expansion**: Modify the pipeline to accommodate data from additional retailers (C, D, E, etc.), enhancing flexibility and scalability. This involves adapting the pipeline to read files from various folders and sources.

2. **Advanced Clustering Techniques**: Experiment with different clustering methods, including graph-based algorithms and others like agglomerative or hierarchical clustering, to improve clustering performance and accuracy.

3. **Dimensionality Reduction Exploration**: Test various dimensionality reduction techniques to optimize the handling of high-dimensional data, potentially improving model performance and interpretability.

4. **Tokenization Optimization**: Enhance the text_tokenization function to include diverse regular expressions and larger, possibly multilingual, language models. This expansion aims to increase the scope of text analysis and ultimately improve clustering results.

5. **TF-IDF Parameter Tuning**: Experiment with different parameters in TfidfVectorizer, such as max_df, min_df, and ngram_range, to refine the TF-IDF representation for better feature extraction.

6. **Similarity Threshold Exploration**: Explore different thresholds in the entity matching process to define product similarities more accurately, ensuring a more nuanced and effective matching process.

7. **Graph Analysis for Clustering**: Investigate the use of graph analysis techniques to evaluate clustering and matching performance, potentially uncovering new insights and optimization opportunities.

8. **Enhanced Clustering Metrics**: Develop better metrics for clustering analysis, moving beyond the current use of unrecognized cluster counts and qualitative inspection, to provide a more robust evaluation of clustering effectiveness.

9. **Pipeline Usability Improvements**: Add detailed instructions on how to run the pipeline in the README.md file, making the project more accessible to new users.

10. **Notebook Enhancement**: Improve the indexing and use of markdown in development notebooks for better documentation and user experience.

These future work items aim to not only enhance the project's existing capabilities but also to expand its scope and applicability in the field of NLP for retail product analysis.

---

This section adds a structured and detailed roadmap for future developments to your README.md file, ensuring clarity of direction and scope for potential contributors and users of the project.

### Usage

To run the project, ensure all dependencies are installed as listed in `requirements.txt`. Use the notebooks in `notebooks-dev/` for a step-by-step approach or execute `run_pipeline.sh` in the `pipeline/` directory to run the entire process.

### Contribution and Feedback

Contributions to the project are welcome. Please refer to the documentation for guidelines on contributing. For feedback and issues, please open an issue in the repository.
