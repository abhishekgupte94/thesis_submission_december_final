import json

with open("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/AVSpeech/AVSpeech_timestamp_json_for_offline_training/AVSpeech_timestamp_json_for_offline_training.json", "r") as f:
    data = json.load(f)

num_unique_keys = len(data)
print(num_unique_keys)
