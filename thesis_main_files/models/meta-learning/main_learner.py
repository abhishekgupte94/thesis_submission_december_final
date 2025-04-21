import torch
import torch.nn as nn
import torch.optim as optim


class MetaLearner(nn.Module):
    def __init__(self, input_shape):
        super(MetaLearner, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def train_meta_learner(X_train, y_train, X_val, y_val, input_shape, device, epochs=50, batch_size=32):
    model = MetaLearner(input_shape).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train).to(device),
                                                   torch.LongTensor(y_train).to(device))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val).to(device),
                                                 torch.LongTensor(y_val).to(device))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    best_val_accuracy = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_accuracy = val_correct / val_total
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_meta_learner.pth')

        print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}')

    return model


def evaluate_meta_learner(model, X_test, y_test, device):
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test).to(device),
                                                  torch.LongTensor(y_test).to(device))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy
