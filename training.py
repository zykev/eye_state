import torch.nn as nn
import argparse
import datetime
from network import LeNet5
from load_data import LoadEyeData
from util import *

parser = argparse.ArgumentParser(description='Eye State Training')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run (default: 50)')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.001)')
parser.add_argument('--batch_size', default=32, type=int, help='number of data in one batch (default: 32)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_TRAIN = 'E:/onlyfat_selfa3D_2/Data/Eye_Images/train_info.csv'
ROOT_TEST = 'E:/onlyfat_selfa3D_2/Data/Eye_Images/test_info.csv'
DATA_DIR = 'E:/onlyfat_selfa3D_2/eye_state/Eye_Images/LeftEyes'
min_loss = 10

def main():
    train_loader, test_loader = LoadEyeData(ROOT_TRAIN, ROOT_TEST, DATA_DIR, args.batch_size)
    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    train_acces = []
    valid_acces = []

    # Train model
    for epoch in range(0, args.epochs):
        # training
        #train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        model, optimizer, train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss.detach().cpu().item())
        train_acces.append(train_acc.detach().cpu().item())

        # validation
        with torch.no_grad():
            #valid_loss, valid_acc = validate(test_loader, model, criterion, device)
            model, valid_loss, valid_acc = validate(test_loader, model, criterion, device)
            valid_losses.append(valid_loss.detach().cpu().item())
            valid_acces.append(valid_acc.detach().cpu().item())

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch:      {epoch}\t'
                  f'Train loss: {train_loss.detach().cpu().item():.4f}\t'
                  f'Valid loss: {valid_loss.detach().cpu().item():.4f}\t'
                  f'Train acc:  {100 * train_acc.detach().cpu().item():.2f}\t'
                  f'Valid acc:  {100 * valid_acc.detach().cpu().item():.2f}')

        # model save
        is_better = valid_loss < min_loss
        if is_better:
            print('better model!')
            min_loss = min(valid_loss, min_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': valid_loss,
                'acc': valid_acc
            })
        else:
            print('Model too bad & not save')

    # training loss and acc plot
    plot_training(train_losses, valid_losses, train_acces, valid_acces)


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    outputs = []
    labels = []

    for data, label in train_loader:
        optimizer.zero_grad()

        data = data.to(device)
        label = label.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, label)
        running_loss += loss.item() * data.size(0)
        outputs.append(output)
        labels.append(label)


        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = get_accuracy(outputs, labels)

    return model, optimizer, epoch_loss, epoch_acc


def validate(test_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0
    outputs = []
    labels = []


    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)

        # Forward pass and record loss
        output = model(data)
        loss = criterion(output, label)
        running_loss += loss.item() * data.size(0)
        outputs.append(output)
        labels.append(label)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = get_accuracy(outputs, labels)

    return model, epoch_loss, epoch_acc


if __name__ == '__main__':
    main()
