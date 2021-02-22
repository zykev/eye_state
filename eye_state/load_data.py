from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import pandas as pd


class EyeData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        eye_data = pd.read_csv(root_dir)
        self.img_name = np.array(eye_data['name'])
        self.img_label = np.array(eye_data['label'])

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name_index = self.img_name[index]
        img_path = os.path.join(self.root_dir, img_name_index)
        img = Image.open(img_path).convert('L')
        label = self.img_label[index]

        if self.transform:
            img = self.transform(img)

        return img, label


def LoadEyeData(root_train, root_test, batch_size):

    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))])
    train_dataset = EyeData(root_train, transform=transform)
    test_dataset = EyeData(root_test, transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, test_loader



