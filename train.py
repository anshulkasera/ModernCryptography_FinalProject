import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_data_loader(X, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, X_train, y_train, epochs=50, learning_rate=0.001):
    train_loader = create_data_loader(X_train, y_train)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {average_loss:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        accuracy = (predictions == y_test).float().mean().item()

    print(f"Test accuracy: {accuracy:.2%}")
    return accuracy