import torch
import numpy as np
from preprocess import preprocess
import torch.optim as optim
from model import LiteNet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in dataloader:
            samples = samples.to(device).float()
            labels = labels.to(device)

            predictions = model(samples)

            if criterion:
                loss = criterion(predictions, labels.long())
                total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0
    return avg_loss, all_preds, all_labels

def test_and_report(model, test_loader, device, class_names):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.
    """
    print("\n--- Starting Final Test ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for samples, labels in test_loader:
            samples = samples.to(device)
            labels = labels.to(device).long()
            
            predictions = model(samples)
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc*100:.2f}%")
    
    print('--- Classification Report ---')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    
    return acc

def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    best_loss = float('inf')
    patience = 5

    for epoch in range(num_epochs):

        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)

        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'Oracle_LiteNet.pth')
            print(f"Model saved to Oracle_LiteNet.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Epoch {epoch+1}/{num_epochs}, best_val={best_loss:.6f}")
                break

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    # inside your training function, after the loop ends (or right before any return)
    model.load_state_dict(torch.load('Oracle_LiteNet.pth', map_location=device))
    model.eval()
    return model




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train = np.load('CICIDS2017/train.npy')
    test = np.load('CICIDS2017/test.npy')
    val  = np.load('CICIDS2017/val.npy')
    print("Finished loading data")

    train_loader, test_loader, val_loader = preprocess(train, test, val, batch_size=128)

    model = LiteNet(num_classes=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    class_names = ['Benign', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'SSH-Patator']
    num_epochs = 100
    model = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs)
    test_and_report(model, test_loader, device, class_names)