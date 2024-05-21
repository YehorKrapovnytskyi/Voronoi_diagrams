import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
import os

class TorchTrainer:
    """
       A trainer class for training and validating a PyTorch model, with support for early stopping and TensorBoard logging.

       This class is designed to accept configuration parameters as strings, allowing for more flexible and user-friendly setup.

       Attributes:
           model (torch.nn.Module): The PyTorch model to be trained.
           train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
           val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
           criterion (str): String name of the loss function (e.g., 'cross_entropy', 'mse').
           optimizer (str): String name of the optimizer (e.g., 'adam', 'sgd').
           scheduler (str, optional): String name of the learning rate scheduler (e.g., 'reduce_lr_on_plateau', 'step_lr'). If None, no scheduler is used.
           device (str, optional): String specifying the device for training (e.g., 'cpu', 'cuda:0'). Defaults to 'cpu'.
           patience (int, optional): Number of epochs to wait for improvement before stopping training. Defaults to 5.

       Methods:
           train(num_epochs): Trains and validates the model for the specified number of epochs.
           train_epoch(): Conducts a single training epoch over the training dataset.
           validate_epoch(): Conducts a validation step over the validation dataset.
           save_checkpoint(filename): Saves the model checkpoint to the specified file.
       """


    def __init__(self, model, train_loader, val_loader, criterion_str, optimizer_str, scheduler_str=None, device_str='cpu', patience=5, debug=False):
        self.model = model.to(device_str)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = self._get_criterion(criterion_str)
        self.optimizer = self._get_optimizer(optimizer_str, model.parameters())
        self.scheduler = self._get_scheduler(scheduler_str, self.optimizer) if scheduler_str else None
        self.device = torch.device(device_str)
        self.patience = patience
        self.writer = SummaryWriter()
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = np.inf
        self.no_improve_epochs = 0
        self.debug = debug
    

    def _get_criterion(self, criterion_str):
        """
        Returns the criterion (loss function) based on the given string.

        Args:
            criterion_str (str): The name of the loss function.

        Returns:
            torch.nn.Module: The loss function.
        """
        if criterion_str == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_str == 'mse':
            return nn.MSELoss()
        elif criterion_str == "binary_cross_entropy":
            return nn.BCELoss()
        # Add more criterion types as needed
        else:
            raise ValueError(f"Unknown criterion type: {criterion_str}")
        
    
    def _get_optimizer(self, optimizer_str, parameters):
        """
        Returns the optimizer based on the given string.

        Args:
            optimizer_str (str): The name of the optimizer.
            parameters (iterable): The parameters to optimize.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if optimizer_str == 'adam':
            return optim.Adam(parameters)
        elif optimizer_str == 'sgd':
            return optim.SGD(parameters, lr=0.01, momentum=0.9)
        # Add more optimizer types as needed
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_str}")
    

    def _get_scheduler(self, scheduler_str, optimizer):
        """
        Returns the learning rate scheduler based on the given string.

        Args:
            scheduler_str (str): The name of the scheduler.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
        """
        if scheduler_str == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(optimizer)
        elif scheduler_str == 'step_lr':
            return StepLR(optimizer, step_size=5, gamma=0.1)
        # Add more scheduler types as needed
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_str}")
        
    
    def train(self, num_epochs):
        """
        Trains and validates the model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.

        Returns:
            torch.nn.Module: The trained model.
        """

        for epoch in range(num_epochs):

            print(f"Epoch {epoch + 1} / {num_epochs}:")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.writer.add_scalar('Training Loss', train_loss, epoch)
            self.writer.add_scalar('Training Accuracy', train_acc, epoch)
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            self.writer.add_scalar('Validation Accuracy', val_acc, epoch)

            print(f"Epoch {epoch + 1} / {num_epochs}:")
            print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")
            print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping logic
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        self.model.load_state_dict(self.best_model_wts)

        return self.model
    
    
    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
        - epoch_loss (float): Average loss for the epoch.
        - epoch_metric (float): Metric for the epoch (accuracy as a percentage for classification, mean absolute error for regression).
        """
        self.model.train()
        running_loss = 0.0
        running_metric = 0.0

        for idx, (inputs, labels) in enumerate(self.train_loader):

            print(f"Batch {idx + 1} / {len(self.train_loader)}")

            #print(labels)

            if self.debug:
                print(f"Training batch input shape: {inputs.shape}, labels shape: {labels.shape}")

            inputs = inputs.to(self.device)  # Use .to(self.device) for data type and device conversion

            if self.model.task_type == 'classification':
                labels = labels.type(torch.LongTensor).to(self.device)
            elif self.model.task_type == 'binary_classification':
                labels = labels.float().to(self.device)  # Use float for BCEWithLogitsLoss
            elif self.model.task_type == 'regression':
                labels = labels.float().to(self.device)  # Use float for regression tasks
            else:
                raise ValueError("task_type must be 'classification', 'binary_classification', or 'regression'")

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if self.model.task_type == 'classification':
                _, preds = torch.max(outputs, 1)
                running_metric += torch.sum(preds == labels).item()
            elif self.model.task_type == 'binary_classification':
                preds = (outputs > 0.5).float()  # Threshold outputs at 0.5
                running_metric += torch.sum(preds == labels).item()
            elif self.model.task_type == 'regression':
                running_metric += torch.sum(torch.abs(outputs - labels)).item()  # Mean absolute error
        
        epoch_loss = running_loss / len(self.train_loader.dataset)

        if self.model.task_type == 'regression':
            epoch_metric = running_metric / len(self.train_loader.dataset)  # Mean absolute error for regression
        else:
            epoch_metric = (running_metric / len(self.train_loader.dataset)) * 100  # Accuracy in percentage for classification

        return epoch_loss, epoch_metric
    
    
    def validate_epoch(self):
        """
        Validate the model for one epoch.

        Returns:
        - total_loss (float): Average loss for the epoch.
        - total_metric (float): Metric for the epoch (accuracy as a percentage for classification, mean absolute error for regression).
        """

        self.model.eval()
        running_loss = 0.0
        running_metric = 0.0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                if self.debug:
                    print(f"Validation batch input shape: {inputs.shape}, labels shape: {labels.shape}")
                
                inputs = inputs.to(self.device)
                
                # Prepare labels based on task type
                if self.model.task_type == 'classification':
                    labels = labels.long().to(self.device)
                elif self.model.task_type == 'binary_classification':
                    labels = labels.float().to(self.device)  # Use float for BCEWithLogitsLoss
                elif self.model.task_type == 'regression':
                    labels = labels.float().to(self.device)  # Use float for regression tasks
                else:
                    raise ValueError("task_type must be 'classification', 'binary_classification', or 'regression'")

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                # Compute metrics
                if self.model.task_type == 'classification':
                    _, preds = torch.max(outputs, 1)
                    running_metric += torch.sum(preds == labels).item()
                elif self.model.task_type == 'binary_classification':
                    preds = (outputs > 0.5).float()
                    running_metric += torch.sum(preds == labels).item()
                elif self.model.task_type == 'regression':
                    running_metric += torch.sum(torch.abs(outputs - labels)).item()  # Sum of absolute errors

        total_loss = running_loss / len(self.val_loader.dataset)

        if self.model.task_type == 'regression':
            total_metric = running_metric / len(self.val_loader.dataset)  # Mean absolute error for regression
        else:
            total_metric = (running_metric / len(self.val_loader.dataset)) * 100  # Accuracy in percentage for classification

        return total_loss, total_metric
    

    def save_checkpoint(self, filename="checkpoint.pth"):
        """
        Saves the model checkpoint to the specified file.

        Args:
            filename (str): The filename for the checkpoint.
        """
        torch.save(self.model.state_dict(), filename)
    