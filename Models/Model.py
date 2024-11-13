import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Model(nn.Module):
    def __init__(self, device, train_loader, val_loader, test_loader):
        super(Model, self).__init__()
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def train_(self, num_epochs, criterion, optimizer):
        self.to(self.device)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            total_correct = 0.0
            total_samples = 0.0

            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for images, labels in self.train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                    # Update progress bar
                    pbar.set_postfix({'Loss': loss.item()})
                    pbar.update(1)

            epoch_loss = running_loss / total_samples
            epoch_acc = total_correct / total_samples
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)

            # Validation
            val_loss, val_acc = self.eval_(criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(
                f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n'
            )
        
    def eval_(self, criterion):
        self.eval()
        running_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = running_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy
    
    def plot_training_history(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Plot Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, label_encoder):
        self.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix on Validation Set')
        plt.show()