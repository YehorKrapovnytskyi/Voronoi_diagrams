from dataloader.VoronoiDataset import *
from model.Resnet_Custom import *
from trainer.TorchTrainer import *

from torch.utils.data import DataLoader, random_split
import torch

DATASET_PATH = '../voronoi_data/voronoi_data.h5'
TRAIN_SHARE = 0.8
VALID_SHARE = 0.1
TEST_SHARE = 0.1
BATCH_SIZE = 256
NUM_INPUT_CHANNELS = 1
NUM_OUTPUTS = 40


dataset_full = VoronoiDataset(DATASET_PATH)
total_size = len(dataset_full)
train_size = int(total_size * TRAIN_SHARE)
valid_size = int(total_size * VALID_SHARE)
test_size = int(total_size * TEST_SHARE)


train_dataset, valid_dataset, test_dataset = random_split(dataset_full, [train_size, valid_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


resnet50_regression = Resnet_Custom(model_name="resnet50", num_input_channels=1, num_outputs=40, pretrained=True, task_type="regression")
torch_trainer = TorchTrainer(model=resnet50_regression, 
                            train_loader=train_loader, 
                            val_loader=valid_loader,
                            criterion_str="mse",
                            optimizer_str="adam",
                            device_str="cpu",
                            )

torch_trainer.train(num_epochs=100)



















