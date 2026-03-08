"""
Helmet Detection - YOLOv8 Training Script
Trains a YOLOv8 model to detect helmets and no-helmets.
"""

from ultralytics import YOLO


def main():
    # Load the pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train the model on the merged dataset
    results = model.train(
        data="data_merged.yaml", # Path to merged dataset config
        epochs=100,              # Number of training epochs
        batch=16,                # Batch size (reduce to 8 if you get CUDA out-of-memory errors)
        imgsz=640,               # Image size
        device=0,                # Use GPU 0 (your RTX 3050)
        workers=4,               # Number of data loading workers
        patience=20,             # Early stopping: stop if no improvement for 20 epochs
        save=True,               # Save checkpoints
        project="runs/train",    # Save results here
        name="helmet_detect",    # Experiment name
        exist_ok=True,           # Overwrite if experiment already exists
        pretrained=True,         # Use pretrained weights
        optimizer="auto",        # Auto-select optimizer
        verbose=True,            # Print detailed logs
    )

    # Print results
    print("\n✅ Training complete!")
    print(f"Best model saved at: runs/train/helmet_detect/weights/best.pt")
    print(f"Last model saved at: runs/train/helmet_detect/weights/last.pt")

    # Validate the best model on the validation set
    print("\n📊 Running validation on best model...")
    best_model = YOLO("runs/train/helmet_detect/weights/best.pt")
    metrics = best_model.val()

    print(f"\nmAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
