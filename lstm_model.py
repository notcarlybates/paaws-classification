"""
Simple LSTM Model for Activity Recognition
Uses the same data as RFT model, but with deep learning instead of handcrafted features.
"""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for activity recognition.

    Architecture:
    - LSTM layers to process temporal sequences
    - Fully connected layers for classification
    """

    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=5, dropout=0.3):
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features (3 for x, y, z accelerometer)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of activity classes to predict
            dropout: Dropout rate
        """
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        # h_n shape: (num_layers, batch_size, hidden_size)
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_size)

        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: LSTM model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda or cpu)

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model.

    Args:
        model: LSTM model
        test_loader: DataLoader for test data
        device: Device (cuda or cpu)

    Returns:
        Tuple of (predictions, true_labels)
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return all_predictions, all_labels
