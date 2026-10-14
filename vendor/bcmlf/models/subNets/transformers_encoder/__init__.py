import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initialize the LinearHead module.

        Args:
            in_features (int): Number of input features (dimension of the hidden representation).
            out_features (int): Number of output features (dimension of the linear head output).
        """
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Forward pass of the LinearHead module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Apply the linear transformation
        return self.linear(x)

