import os
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from utils import split_coco_dataset, setup_cfg

def register_voronoi_datasets(train_dataset_path, val_dataset_path):
    # Register the training set
    register_coco_instances("voronoi_train_dataset", {}, 
                            os.path.join(train_dataset_path, "annotations.json"),
                            os.path.join(train_dataset_path, "images"))
    
    # Register the validation set
    register_coco_instances("voronoi_val_dataset", {}, 
                            os.path.join(val_dataset_path, "annotations.json"),
                            os.path.join(val_dataset_path, "images"))


DATASET_BASE = "../voronoi_dataset"
JSON_PATH = os.path.join(DATASET_BASE, "full", "annotations.json")
IMAGES_PATH = os.path.join(DATASET_BASE, "full")
SPLIT_PATH = os.path.join(DATASET_BASE, "split")

split_coco_dataset(JSON_PATH, IMAGES_PATH, SPLIT_PATH)

TRAIN_DATASET_PATH = os.path.join(DATASET_BASE, "split", "train")
VAL_DATASET_PATH = os.path.join(DATASET_BASE, "split", "val")

register_voronoi_datasets(TRAIN_DATASET_PATH, VAL_DATASET_PATH)

cfg = setup_cfg(
    base_config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    output_dir="./output",
    train_dataset="voronoi_train_dataset",
    val_dataset="voronoi_val_dataset",
    weights=model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"),
    num_classes=1,
    ims_per_batch=2,
    base_lr=0.01,
    max_iter=100,
    batch_size_per_image=128,
    num_workers=2,
    eval_period=10,
    device="gpu"
)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save the final model weights
final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print(f"Model saved to {final_model_path}")
