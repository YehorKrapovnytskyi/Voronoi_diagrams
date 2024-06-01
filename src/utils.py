import os
import json
import shutil
from sklearn.model_selection import train_test_split

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

