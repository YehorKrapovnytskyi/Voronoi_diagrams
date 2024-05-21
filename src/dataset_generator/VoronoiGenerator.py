import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
import h5py


class VoronoiGenerator:
    """
    A class to generate and save Voronoi diagrams as images and their corresponding points data.

    Attributes:
        DEFAULT_POINTS (int): Default number of points to generate Voronoi diagrams.
        DEFAULT_DIMENSION (int): Default height and width for the Voronoi diagrams.
    """
    DEFAULT_POINTS = 20
    DEFAULT_DIMENSION = 224

    
    def __init__(self, save_directory):
        """
        Initialize the generator with a directory to save images and data.

        Args:
            save_directory (str): Directory path to save generated Voronoi diagrams and data.
        """
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)
        self.hdf5_path = os.path.join(save_directory, 'voronoi_data.h5')  # Specify the HDF5 file name


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
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=True, point_size=0.75, line_colors='black', line_width=1)
        self._set_line_styles(ax)
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.axis('off')
        
        # Save plot to numpy array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Convert an image to grayscale
        image = np.mean(image, axis=2).astype(np.uint8)[..., np.newaxis]

        return image, points
    

    def _set_line_styles(self, ax):
        """Set line styles for Voronoi plot to solid."""
        for collection in ax.collections:
            collection.set_linestyle('solid')


    def generate_voronoi_dataset(self, N_images, N_points=None, height=None, width=None):
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
        
        with h5py.File(self.hdf5_path, 'w') as f:
            img_dset = f.create_dataset('images', (N_images, height, width, 1), dtype='uint8')
            pts_dset = f.create_dataset('points', (N_images, N_points, 2), dtype='float32')

            for image_num in range(N_images):
                image, points = self.generate_voronoi_diagram(N_points, height, width)
                img_dset[image_num] = image
                pts_dset[image_num] = points

                if (image_num + 1) % max(N_images // 10, 1) == 0 or image_num == N_images - 1:
                    percentage_complete = (image_num + 1) / N_images * 100
                    print(f"Progress: {percentage_complete:.0f}% - Dataset saved at image {image_num + 1}")