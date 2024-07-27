from ultralytics import YOLO
import os
import cv2


model = YOLO('/Users/akshitasachdeva/Desktop/yolov8/runs/detect/train10/weights/best.pt')


test_images_dir = '/Users/akshitasachdeva/Desktop/runs/tests/60set-test'
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]


results_dir = '/Users/akshitasachdeva/Desktop/runs/results/60set-15e'
os.makedirs(results_dir, exist_ok=True)

results = model(test_images)


for i, result in enumerate(results):
   
    print(f"Results for {test_images[i]}:")
    print(result)

   
    img = cv2.imread(result.path)

    
    boxes = result.boxes.xyxy
    labels = result.boxes.cls
    names = result.names

   
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        class_name = names[int(label)]
        color = (0, 255, 0)  
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    
    save_path = os.path.join(results_dir, os.path.basename(test_images[i]))
    cv2.imwrite(save_path, img)
    print(f"Saved result for {test_images[i]} to {save_path}")

    
    cv2.imshow('Result', img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()


print(f"Contents of {results_dir}:")
print(os.listdir(results_dir))

