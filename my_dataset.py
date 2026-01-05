from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                if img.mode == 'L':
                    img = img.convert('RGB')
                elif img.mode == 'RGBA':
                    img = img.convert('RGB')
                else:
                    raise ValueError(f"Unsupported image mode: {img.mode} for {img_path}")
            label = self.images_class[item]
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224), -1
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class WeightedDataset(MyDataSet):#加权数据集
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, self.weights[index]

