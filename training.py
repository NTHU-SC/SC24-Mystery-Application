from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11x.pt") 

# Train the model
model.train(
    data="/home/evanpai/mystery_app/working/cat_dataset.yaml",  # Path to YAML file
    epochs=25,  # Number of epochs
    imgsz=640,  # Image size
    batch=8,   # Batch size
    workers=8,  # Number of data loader workers
    freeze=20,
    lr0=1e-4,
    device=0,    # GPU device (set 'cpu' for CPU training)
    project="/home/evanpai/mystery_app/yolo_training",  # Custom output directory
    name="cat_detection",  # Subdirectory for this specific run
    classes=[0]  # Include only the "Cats" class (class ID = 0)
)