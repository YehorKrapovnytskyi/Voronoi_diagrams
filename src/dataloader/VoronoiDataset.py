import torch
from torch.utils.data import Dataset
import h5py

class VoronoiDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the HDF5 file with image and points data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_path = file_path
        self.transform = transform
        
        # Open the HDF5 file and get references to the datasets
        with h5py.File(self.file_path, 'r') as file:
            self.len = file['images'].shape[0]
            # Keeping these in memory might be necessary depending on the size of the data
            # If data is too large, consider reading batches or single items per iteration
            self.image_shape = file['images'].shape[1:]  # Capture image shape info for transforms


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
        points = torch.from_numpy(points).float().flatten() # flatten the coordinates for regression

        if self.transform:
            image = self.transform(image)

        sample = (image, points)
        return sample

