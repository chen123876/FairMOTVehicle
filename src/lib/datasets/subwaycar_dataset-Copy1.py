import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

class SubwayCarDataset(Dataset):
    def __init__(self, data_cfg, split='train', img_size=(608,1088), transforms=None, augment=False):
        super(SubwayCarDataset, self).__init__()
        self.split = split
        self.img_size = img_size  # (H, W)
        self.transforms = transforms
        self.augment = augment

        # 读取 JSON 配置
        with open(data_cfg, 'r') as f:
            cfg = json.load(f)
        self.root = cfg['root']

        # train / val 文件名，只保留文件名
        self.data_list = os.path.basename(cfg[split]['SubwayCar'])

        # 拼接完整路径
        list_path = os.path.join(self.root, self.data_list)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"{list_path} 不存在，请检查路径")

        # 读取图片路径
        with open(list_path, 'r') as f:
            self.img_paths = [line.strip() for line in f]

        # 自动统计车辆 ID (nID)
        self.nID = 0
        for img_path in self.img_paths:
            label_file = os.path.join(
                self.root, 'SubwayCar/labels_with_ids', self.split, os.path.basename(img_path).replace('.jpg','.txt'))
            if os.path.exists(label_file):
                with open(label_file,'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            vid = int(parts[0])
                            if vid + 1 > self.nID:
                                self.nID = vid + 1

        self.num_classes = cfg['num_classes']
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.default_resolution = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_paths[index])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size[1], self.img_size[0]))

        if self.transforms:
            img = self.transforms(img)  # -> (C,H,W)
        else:
            img = T.ToTensor()(img)

        # 返回 dict，符合 FairMOT 期望
        target = {
            'input': img,
            'labels': torch.tensor([0], dtype=torch.long)  # 占位 label
        }
        return target


# ------------------ collate_fn ------------------
def subwaycar_collate_fn(batch):
    # batch 是 list of dict
    inputs = torch.stack([b['input'] for b in batch], dim=0)   # -> (B,C,H,W)
    labels = torch.stack([b['labels'] for b in batch], dim=0)  # -> (B,1)
    return {'input': inputs, 'labels': labels}


# ------------------ 测试 DataLoader ------------------
if __name__ == "__main__":
    data_cfg = "./src/lib/datasets/subwaycar.json"  # 替换你的路径
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.408, 0.447, 0.47],
                    std=[0.289, 0.274, 0.278])
    ])
    dataset = SubwayCarDataset(
        data_cfg=data_cfg,
        split='train',
        img_size=(608,1088),
        transforms=transforms,
        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=subwaycar_collate_fn  # 关键！
    )

    for batch in loader:
        imgs = batch['input']   # shape: (B, C, H, W)
        labels = batch['labels']  # shape: (B,1)
        print("Batch images shape:", imgs.shape)
        print("Batch labels shape:", labels.shape)
        break
