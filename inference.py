from ultralytics import YOLO
import os
from PIL import Image, ImageDraw

# Directory containing images
image_dir = "/home/evanpai/mystery_app/benchmarks_v2/train/images"
output_dir = "/home/evanpai/mystery_app/outputs_yolo"
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv8 model
# model = YOLO("/home/evanpai/mystery_app/yolo_training/cat_detection2/weights/best.pt")
model = YOLO("yolo11x.pt")  

# Retrieve class names from the model
class_names = model.names  # A list of class names

# Function to draw bounding boxes for specific target classes
def draw_target_boxes(image, boxes, scores, labels, class_names, target_classes, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold and class_names[int(label)] in target_classes:
            x1, y1, x2, y2 = box
            label_name = class_names[int(label)]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{label_name} ({score:.2f})", fill="red")
    return image

# Target classes (e.g., cat and dog)
target_classes = ["cat"]

# Process all images in the directory
print("Starting inference...")
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        
        # Run inference
        results = model(image_path)

        for result in results:  # Iterate over results
            # Extract boxes, confidence scores, and class IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            labels = result.boxes.cls.cpu().numpy()  # Class IDs

            # Open the image and draw target class boxes
            image = Image.open(image_path).convert("RGB")
            annotated_image = draw_target_boxes(image, boxes, scores, labels, class_names, target_classes, threshold=0.5)

            # Save the annotated image
            annotated_image.save(os.path.join(output_dir, filename))
            print(f"Processed {filename}")

print("Inference completed. Results saved to:", output_dir)
