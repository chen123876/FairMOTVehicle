from torch.utils.data import Dataset
from PIL import Image
import os

class SubwayCarDataset(Dataset):
    def __init__(self, root, train_file, transforms=None):
        # train_file 是 subwaycar.train 文件
        with open(train_file, 'r') as f:
            self.img_paths = [os.path.join(root, line.strip()) for line in f.readlines()]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img

