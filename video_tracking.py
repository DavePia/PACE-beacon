from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import videointelligence as vi
import io, json

def setup_client():    
    load_dotenv()
    client = vi.VideoIntelligenceServiceClient()
    return client

def analyze_video(video_path: str, mode: str):
    
    client = setup_client()

    with io.open(video_path, "rb") as f:
        input_content = f.read()

    if mode == "labels":
        features = [vi.Feature.LABEL_DETECTION]
    elif mode == "objects":
        features = [vi.Feature.OBJECT_TRACKING]
    else:
        raise ValueError("mode must be either 'labels' or 'objects'")

    operation = client.annotate_video(
        request={"features": features, "input_content": input_content}
    )
    result = operation.result(timeout=600)
    annotations = result.annotation_results[0]

    if mode == "labels":
        tags = sorted({l.entity.description for l in annotations.segment_label_annotations})
        output = {"video_file": video_path, "tags": tags}
        with open("label_results.json", "w") as f:
            json.dump(output, f, indent=2)
    else:
        objects = []
        for o in annotations.object_annotations:
            objects.append({
                "description": o.entity.description,
                "confidence": round(o.confidence * 100, 2),
                "start": o.segment.start_time_offset.seconds + o.segment.start_time_offset.microseconds / 1e6,
                "end": o.segment.end_time_offset.seconds + o.segment.end_time_offset.microseconds / 1e6,
            })
        output = {"video_file": video_path, "objects": objects}
        with open("object_results.json", "w") as f:
            json.dump(output, f, indent=2)


analyze_video("videos/sample.mp4", "labels")
analyze_video("videos/sample.mp4", "objects")
    

