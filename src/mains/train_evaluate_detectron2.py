import os
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances


import os
from utils import split_coco_dataset


def register_voronoi_datasets(train_dataset_path, val_dataset_path, test_dataset_path):
    # Register the training set
    register_coco_instances("voronoi_train_dataset", 
                            {}, 
                            os.path.join(train_dataset_path, "annotations.json"),
                            os.path.join(train_dataset_path, "images")
    )
    
    # Register the validation set
    register_coco_instances("voronoi_val_dataset", 
                            {}, 
                            os.path.join(val_dataset_path, "annotations.json"),
                            os.path.join(val_dataset_path, "images")
    )

    # Register the test set
    register_coco_instances("voronoi_test_dataset", 
                            {}, 
                            os.path.join(test_dataset_path, "annotations.json"),
                            os.path.join(test_dataset_path, "images")
    )


DATASET_BASE = "../voronoi_dataset"
JSON_PATH = os.path.join(DATASET_BASE, "full", "annotations.json")
IMAGES_PATH = os.path.join(DATASET_BASE, "full")
SPLIT_PATH = os.path.join(DATASET_BASE, "split")

split_coco_dataset(JSON_PATH, IMAGES_PATH, SPLIT_PATH)

TRAIN_DATASET_PATH = os.path.join(DATASET_BASE, "split", "train")
VAL_DATASET_PATH = os.path.join(DATASET_BASE, "split", "val")
TEST_DATASET_PATH = os.path.join(DATASET_BASE, "split", "test")


register_voronoi_datasets(TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("voronoi_train_dataset",)
cfg.DATASETS.TEST = ("voronoi_val_dataset",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Load pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 100  # Adjust the number of iterations based on your dataset size
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (voronoi_point)

cfg.MODEL.DEVICE = "cpu"

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


