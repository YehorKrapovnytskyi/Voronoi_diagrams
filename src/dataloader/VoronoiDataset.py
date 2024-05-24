import torch
from torch.utils.data import Dataset
import h5py

class VoronoiDataset(Dataset):
    def __init__(self, file_path, transform=None, box_size=3):
        """
        Args:
            file_path (string): Path to the HDF5 file with image and points data.
            transform (callable, optional): Optional transform to be applied on a sample.
            box_size (int): The size of the bounding box around each point.
        """

        self.file_path = file_path
        self.transform = transform
        self.box_size = box_size
        
        # Open the HDF5 file and get references to the datasets
        with h5py.File(self.file_path, 'r') as file:
            self.len = file['images'].shape[0]
            self.image_shape = file['images'].shape[1:]


    def __len__(self):
        return self.len
    

    def __getitem__(self, idx):
        # Access the HDF5 file for every individual item fetch
        with h5py.File(self.file_path, 'r') as file:
            image = file['images'][idx]
            points = file['points'][idx]

        
        # Convert data to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Change to C, H, W format
        image = image / 255.0 # scale the pixels
        
        # Create bounding boxes around each point
        boxes = []
        for point in points:
            x_center, y_center = point
            xmin = x_center - self.box_size / 2
            ymin = y_center - self.box_size / 2
            xmax = x_center + self.box_size / 2
            ymax = y_center + self.box_size / 2
            boxes.append([xmin, ymin, xmax, ymax])
        

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((points.shape[0],), dtype=torch.int64)  # Voronoi points have label 1

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target
        

