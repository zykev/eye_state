import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_checkpoint(state):
    if not os.path.exists('./model'):
        os.makedirs('./model')

    save_dir = './model/epoch'+str(state['epoch']) + '_loss_' + str(round(float(state['loss']), 3)) + '_acc_' + str(round(float(state['acc']), 3))
    torch.save(state, save_dir, _use_new_zipfile_serialization=False)

def get_accuracy(outputs, targets):
    '''
     Function for computing the accuracy of the predictions over the entire data_loader
    '''
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def plot_training(train_losses, valid_losses, train_acces, valid_acces):
    '''
    Function for plotting training and validation losses/ acc
    '''

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    train_acces = np.array(train_acces)
    valid_acces = np.array(valid_acces)

    plt.figure(figsize=(16, 5))
    plt.subplot(121)

    plt.plot(train_losses, color='blue', label='Training loss')
    plt.plot(valid_losses, color='red', label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')

    plt.subplot(122)
    plt.plot(train_acces, color='blue', label='Training acc')
    plt.plot(valid_acces, color='red', label='Validation acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Acc over epochs')

    plt.savefig('./training_plot.jpg')


