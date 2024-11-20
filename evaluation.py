from ultralytics import YOLO

# Load the trained model
model = YOLO("/home/evanpai/mystery_app/yolo_training/cat_detection6/weights/best.pt")
#model = YOLO("yolo11x.pt")  

# Run validation on the dataset
metrics = model.val(
    data="/home/evanpai/mystery_app/working/cat_dataset.yaml",  # Path to dataset YAML file
    split='test',  # Use the test set instead of the validation set
    imgsz=640,  # Image size
    batch=16,    # Batch size
    project="/home/evanpai/mystery_app/yolo_evaluation",  # Custom output directory
    name="cat_detection",  # Subdirectory for this specific run
    device=0,    # GPU device (set 'cpu' for CPU validation)
    classes=[0]
)

# Example: Access specific metrics
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
mAP50 = metrics.results_dict['metrics/mAP50(B)']
mAP50_95 = metrics.results_dict['metrics/mAP50-95(B)']

# Print specific metrics
print("\nSpecific Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"mAP50: {mAP50:.2f}")
print(f"mAP50-95: {mAP50_95:.2f}")