#!/bin/bash

# Run data_ingestion.py
echo "Starting data ingestion..."
python src/data_extraction_processing.py --folder_path="$folder_path"

# Run entity_clustering.py
echo "Starting entity clustering..."
python src/entity_clustering.py --input_file="$raw_data_file" --output_file="$processed_data_file"

# Run entity_matching.py
echo "Starting entity matching..."
python src/entity_matching.py --input_file="$processed_data_file" --model_file="$model_file"

echo "Pipeline completed."
