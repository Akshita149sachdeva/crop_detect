import torch
from ultralytics import YOLO

model = YOLO('/Users/akshitasachdeva/Desktop/yolov8/runs/detect/train5/weights')  
test_images_path = 'path/to/test/images'

# Run inference on the test images
results = model.predict(source=test_images_path, save=True)

# Print results for each test image
for result in results:
    print(result)

# If you have a test dataset configuration file, you can evaluate the model
test_data_yaml = 'path/to/your/test_data.yaml'  # Update with the path to your test data configuration file

# Evaluate the model on the test dataset
metrics = model.val(data=test_data_yaml)

# Print evaluation metrics
print("Evaluation Metrics:")
print(metrics)
