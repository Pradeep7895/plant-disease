import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformerHybrid(nn.Module):
    def __init__(self):
        super(CNNTransformerHybrid, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Transformer encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32*56*56, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)

        self.fc1 = nn.Linear(32*56*56, 256)  # Adjust for your input size
        self.fc2 = nn.Linear(256, 10)  # Assuming 10 classes for diseases

    def forward(self, x):
        # CNN part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the transformer
        x = x.view(x.size(0), -1)

        # Transformer part
        x = self.transformer_encoder(x.unsqueeze(1))  # Add sequence dimension

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)

        return x

# Example usage
if __name__ == '__main__':
    model = CNNTransformerHybrid()
    print(model)