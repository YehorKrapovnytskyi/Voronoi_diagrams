import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
import h5py
from PIL import Image
import json


class VoronoiGenerator:
    """
    A class to generate and save Voronoi diagrams as images and their corresponding points data.

    Attributes:
        DEFAULT_POINTS (int): Default number of points to generate Voronoi diagrams.
        DEFAULT_DIMENSION (int): Default height and width for the Voronoi diagrams.
    """
    DEFAULT_POINTS = 20
    DEFAULT_DIMENSION = 800
    BBOX_WIDTH = 2
    BBOX_HEIGHT = 2

    
    def __init__(self, save_directory, save_format="hdf5"):
        """
        Initialize the generator with a directory to save images and data.

        Args:
            save_directory (str): Directory path to save generated Voronoi diagrams and data.
            save_format (str): Format to save data ('hdf5' or 'coco'). Default is 'hdf5'.
        """
        self.save_directory = save_directory
        self.save_format = save_format
        os.makedirs(save_directory, exist_ok=True)
        self.hdf5_path = os.path.join(save_directory, 'voronoi_dataset.h5')  # Specify the HDF5 file name
        self.coco_annotations = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "voronoi_point"}]
        }


    def generate_voronoi_diagram(self, N_points, height, width):
        """
        Generate a single Voronoi diagram with specified parameters and return it as a numpy array.

        Args:
            N_points (int): Number of points to generate the Voronoi diagram.
            height (int): Height of the output image.
            width (int): Width of the output image.

        Returns:
            tuple: A tuple containing the generated image as a numpy array and the points used for the Voronoi diagram.
        """
        points = np.round(np.random.rand(N_points, 2) * np.array([width, height])).astype(int)
        vor = Voronoi(points)

        plt.ioff()
        figsize = (width / self.DEFAULT_DIMENSION, height / self.DEFAULT_DIMENSION) # Convert pixels to inches
        fig, ax = plt.subplots(figsize=figsize, dpi=self.DEFAULT_DIMENSION)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors='black', line_width=0.3)
        ax.plot(points[:, 0], points[:, 1], color='red', marker='o', ms=0.2, lw=0, linestyle="", mew=0)
        self._set_line_styles(ax)

        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.axis('off')
        
        # Save plot to numpy array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return image, points
    

    def _set_line_styles(self, ax):
        """Set line styles for Voronoi plot to solid."""
        for collection in ax.collections:
            collection.set_linestyle('solid')


    def generate_voronoi_dataset(self, N_images, N_points=None, height=None, width=None, bbox_height=None, bbox_width=None):
        """
        Generate and save multiple Voronoi images with specified settings.

        Args:
            N_images (int): Number of Voronoi images to generate.
            N_points (int, optional): Number of points for each Voronoi diagram. Defaults to DEFAULT_POINTS.
            height (int, optional): Height of each Voronoi image. Defaults to DEFAULT_DIMENSION.
            width (int, optional): Width of each Voronoi image. Defaults to DEFAULT_DIMENSION.
        """
        N_points = N_points or self.DEFAULT_POINTS
        height = height or self.DEFAULT_DIMENSION
        width = width or self.DEFAULT_DIMENSION
        
        bbox_height = bbox_height or self.BBOX_HEIGHT
        bbox_width = bbox_width or self.BBOX_WIDTH

        if self.save_format == 'hdf5':
        
            with h5py.File(self.hdf5_path, 'w') as f:
                img_dset = f.create_dataset('images', (N_images, height, width, 3), dtype='uint8')
                pts_dset = f.create_dataset('points', (N_images, N_points, 2), dtype='float32')

                for image_num in range(N_images):
                    image, points = self.generate_voronoi_diagram(N_points, height, width)
                    img_dset[image_num] = image
                    pts_dset[image_num] = points

                    if (image_num + 1) % max(N_images // 10, 1) == 0 or image_num == N_images - 1:
                        percentage_complete = (image_num + 1) / N_images * 100
                        print(f"Progress: {percentage_complete:.0f}% - Dataset saved at image {image_num + 1}")
        
        elif self.save_format == 'coco':
            img_id = 0
            ann_id = 0

            for image_num in range(N_images):
                image, points = self.generate_voronoi_diagram(N_points, height, width)
                img_filename = os.path.join(self.save_directory, f"image_{image_num:05d}.png")
                Image.fromarray(image).save(img_filename)

                self.coco_annotations["images"].append({
                    "id": int(img_id),
                    "width": int(width),
                    "height": int(height),
                    "file_name": os.path.basename(img_filename)
                })

                for point in points:
                    x, y = point
                    bbox = [int(x), int(y), int(bbox_width), int(bbox_height)] 
                    self.coco_annotations["annotations"].append({
                        "id": int(ann_id),
                        "image_id": int(img_id),
                        "category_id": 1,
                        "bbox": bbox,
                        "area": bbox_height * bbox_width,
                        "iscrowd": 0
                    })
                    ann_id += 1

                img_id += 1

                if (image_num + 1) % max(N_images // 10, 1) == 0 or image_num == N_images - 1:
                    percentage_complete = (image_num + 1) / N_images * 100
                    print(f"Progress: {percentage_complete:.0f}% - Dataset saved at image {image_num + 1}")

            # Save COCO annotations
            with open(os.path.join(self.save_directory, 'annotations.json'), 'w') as json_file:
                json.dump(self.coco_annotations, json_file, indent=4)