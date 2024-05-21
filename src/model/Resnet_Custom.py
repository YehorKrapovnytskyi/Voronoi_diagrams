import torch
import torch.nn as nn
import torchvision.models as models

class Resnet_Custom(nn.Module):
    """
    A ResNet-based neural network for regression tasks with support for various ResNet architectures,
    input channels and number of outputs.

    Args:
        model_name (str): Name of the ResNet model to use. Options are 'resnet18', 'resnet34', 'resnet50',
                          'resnet101', and 'resnet152'. Default is 'resnet50'.
        num_input_channels (int): Number of input channels for the model. Default is 1 (grayscale images).
        num_outputs (int): Number of regression outputs. Default is 20.
        pretrained (bool): If True, use a model pre-trained on ImageNet. Default is True.
        task_type (str): Type of task ('classification', 'binary_classification', or 'regression'). Default is 'classification'.

    Attributes:
        resnet (torchvision.models.ResNet): The chosen ResNet model with modifications.
        task_type (str): The type of task, which determines the activation function for the output layer.
    """
    def __init__(self, model_name="resnet50", num_input_channels=1, num_outputs=20, pretrained=True, task_type='classification'):
        super(Resnet_Custom, self).__init__()

        self.task_type = task_type

        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }

        if model_name not in model_dict:
            raise ValueError(f"Invalid model name. Available options are: {list(model_dict.keys())}")

        self.resnet = model_dict[model_name](pretrained=pretrained)


        # Modify the first convolution layer of ResNet to account for channel number
        self.resnet.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained and num_input_channels == 1:
            # Copy weights from the pretrained model's first conv layer (average across the channels if num_channels is 1)
            with torch.no_grad():
                self.resnet.conv1.weight[:, 0] = self.resnet.conv1.weight.mean(dim=1) # Average weights across all channels of every filter
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_outputs)

        if self.task_type == 'classification':
            self.activation = nn.Softmax(dim=1)  # Use Softmax for multi-class classification. Dim 1 assuming input in (batch_size, num_classes) format
        elif self.task_type == 'binary_classification':
            self.activation = nn.Sigmoid()  # Use Sigmoid for binary classification
        elif self.task_type == 'regression':
            self.activation = nn.Identity()  # No activation function for regression
        else:
            raise ValueError("task_type must be 'classification', 'binary_classification', or 'regression'")


    def forward(self, x):
        x = self.resnet(x)
        x = self.activation(x)
        return x
