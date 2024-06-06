import os
import json
import shutil
from sklearn.model_selection import train_test_split
from detectron2.config import get_cfg
from detectron2 import model_zoo

def split_coco_dataset(json_path : str, image_dir : str, output_dir : str, train_size=0.8, val_size=0.1, test_size=0.1):
    with open(json_path) as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']

    train_images, temp_images = train_test_split(images, train_size=train_size, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_size/(val_size + test_size), random_state=42)

    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    os.makedirs(output_dir, exist_ok=True)

    for split, split_images in splits.items():
        split_annotations = [anno for anno in annotations if anno['image_id'] in [img['id'] for img in split_images]]

        split_coco = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': coco['categories']
        }

        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        with open(os.path.join(split_dir, 'annotations.json'), 'w') as f:
            json.dump(split_coco, f)

        split_image_dir = os.path.join(split_dir, 'images')
        os.makedirs(split_image_dir, exist_ok=True)

        for image in split_images:
            image_path = os.path.join(image_dir, image['file_name'])
            shutil.copy(image_path, split_image_dir)


def setup_cfg(base_config_path, output_dir, train_dataset, val_dataset, test_dataset=None, weights=None, num_classes=1, 
              ims_per_batch=2, base_lr=0.01, max_iter=100, batch_size_per_image=128, num_workers=2, eval_period=10, device='gpu'):
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_config_path))
    
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    if test_dataset:
        cfg.DATASETS.TEST = (test_dataset,)
        
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = weights if weights else model_zoo.get_checkpoint_url(base_config_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


