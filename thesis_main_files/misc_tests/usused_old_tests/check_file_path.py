import os
import pandas as pd

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
print(current_script_dir)
###### An example of relative path usage

#
# import os
# import pandas as pd
#
# # Get the directory of the current script
# current_script_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Compute the relative path to the CSV file
# metadata_relative_path = os.path.join(
#     current_script_dir,
#     "../../data/processed_files/csv/lav_df/metadata/metadata.csv"
# )
#
# # Resolve the absolute path for safety
# metadata_path = os.path.abspath(metadata_relative_path)
#
# # Load the CSV file
# data = pd.read_csv(metadata_path)
#
# # Print the resolved path for debugging
# print(f"Metadata Path: {metadata_path}")
