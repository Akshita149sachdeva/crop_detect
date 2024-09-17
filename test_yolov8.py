import torch
from ultralytics import YOLO

model = YOLO('/Users/akshitasachdeva/Desktop/yolov8/runs/detect/train5/weights')  
test_images_path = 'path/to/test/images'


results = model.predict(source=test_images_path, save=True)


for result in results:
    print(result)


test_data_yaml = 'path/to/your/test_data.yaml'  # Update with the path to your test data configuration file


metrics = model.val(data=test_data_yaml)


print("Evaluation Metrics:")
print(metrics)
