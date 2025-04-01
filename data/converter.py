from loader import load_dataset
from camera import Camera
from pathlib import Path
import sys
import json

if len(sys.argv) != 4:
    print("Usage: %s input_folder input_dataset output_json" % sys.argv[0])
    exit(1)

input_folder, input_dataset, output_json = sys.argv[1:]

train_cameras, test_cameras, scene_info = load_dataset(
    Path(input_folder),
    input_dataset,
    eval=True)

def cam2json(camera:Camera):
    return {
        "image_name": camera.image_name,
        "view":       camera.world_view_transform.tolist(),
        "projection": camera.projection_matrix.tolist()
    }

data = {
    "train_cameras": [ cam2json(c) for c in train_cameras ],
    "test_cameras":  [ cam2json(c) for c in test_cameras ],
    "points":  scene_info.point_cloud.points.tolist(),
    "colors":  scene_info.point_cloud.colors.tolist(),
    "normals": scene_info.point_cloud.normals.tolist()
}

with open(output_json, 'w') as f:
    json.dump(data, f)