import os
import json
# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metadata_dir = os.path.join(BASE_DIR, 'dataset/metadata')

lav_matadata = os.path.join(metadata_dir, 'lav_df', 'metadata.json')
lav_matadata_min = os.path.join(metadata_dir, 'lav_df', 'metadata.min.json')


