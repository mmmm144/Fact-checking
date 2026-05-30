import os
import json

# Define the path to the output.json file
output_file = "output.json"

# Load the JSON data from output.json
with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Iterate through the JSON data and remove .mp4 files from the media attribute
for entry in data:
    if "media" in entry and isinstance(entry["media"], list):
        entry["media"] = [media_file for media_file in entry["media"] if not media_file.endswith(".mp4")]

# Save the updated JSON data back to output.json
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Cleanup complete. Removed .mp4 files from output.json.")