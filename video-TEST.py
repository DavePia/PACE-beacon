from dotenv import load_dotenv
import json
from google.cloud import videointelligence_v1 as vi
from google.cloud.videointelligence_v1.types import AnnotateVideoResponse
from typing import cast

# 1. Load credentials from .env
load_dotenv()

# 2. Initialize the Video Intelligence client
client = vi.VideoIntelligenceServiceClient()

# 3. Path to your local video file
local_path = "videos/sample.mp4"

print(f"Analyzing local video: {local_path}")

# 4. Read the video file as bytes
with open(local_path, "rb") as file:
    input_content = file.read()

# 5. Request label detection (object and scene tagging)
features = [vi.Feature.LABEL_DETECTION]

operation = client.annotate_video(
    request={"features": features, "input_content": input_content}
)

print("Processing video... this may take a few minutes.")

# 6. Wait for results
result = operation.result(timeout=600)

if result is None:
    raise RuntimeError("API returned None unexpectedly")

result_typed = cast(AnnotateVideoResponse, result)
annotations = result_typed.annotation_results[0]

# 7. Extract tags
tags = sorted(list({label.entity.description for label in annotations.segment_label_annotations}))

# 8. Save results to JSON
profile = {
    "video_file": local_path,
    "tag_count": len(tags),
    "tags": tags
}

with open("video_profile.json", "w") as f:
    json.dump(profile, f, indent=2)

print("Sample tags:", tags[:10])
