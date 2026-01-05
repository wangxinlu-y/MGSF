import os
import math
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ✅ 使用你工程里的 DeiT 定义
from models import deit_base_patch16_224


class WeightCalculatorDeiT:
    def __init__(self,
                 model_path,
                 data_path,
                 num_classes=14,
                 device='cuda:0'):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.data_path = data_path
        self.model = deit_base_patch16_224(pretrained=False, num_classes=num_classes)

        assert os.path.exists(model_path), f"model_path: {model_path} 不存在！"
        print(f"Loading best linear DeiT model from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        msg = self.model.load_state_dict(state_dict, strict=True)
        print("Load state dict result:", msg)

        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def calculate_weights(self,
                          T1=2.0,
                          T2=1.5,
                          alpha=-0.2,
                          beta=-0.05):
        train_dir = os.path.join(self.data_path, "train")
        assert os.path.exists(train_dir)

        dataset = ImageFolder(root=train_dir, transform=self.transform)
        loader = DataLoader(dataset,
                            batch_size=512,
                            shuffle=False,
                            drop_last=False)

        weights = []
        all_labels = []
        all_probabilities = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                entropy = -torch.sum(
                    probabilities * torch.log(probabilities + 1e-10),
                    dim=1
                )
                preds = torch.argmax(probabilities, dim=1)
                correct_mask = (preds == labels)

                for i in range(len(labels)):
                    ei = entropy[i].item()
                    if correct_mask[i]:
                        wi = 1 - math.exp((-ei / T1) + alpha)
                    else:
                        wi = 1 + math.exp((-ei / T2) + beta)
                    weights.append(wi)

        all_probabilities = np.concatenate(all_probabilities)
        all_labels = np.concatenate(all_labels)

        return np.array(weights), all_labels, all_probabilities

    def save_weights(self,
                     save_path='new_weights_deit.npz',
                     **kwargs):
        weights, labels, probs = self.calculate_weights(**kwargs)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        np.savez(save_path,
                 weights=weights,
                 labels=labels,
                 probabilities=probs)

        print(f"save: {save_path}")
        print(f"weight: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")


if __name__ == '__main__':
    calculator = WeightCalculatorDeiT(
        model_path='/media/magneto/disk/wxl/MGSF_TOPK/DEiT/linear_pth/NWPU20_best_linear_deit.pth',
        data_path=r"/media/magneto/disk/wxl/dataset/NWPU4520_percent",
        num_classes=45,
        device='cuda:0'
    )
    calculator.save_weights(
        save_path='/media/magneto/disk/wxl/MGSF_TOPK/DEiT/new_weights',
        T1=2.0,
        T2=1.5,
        alpha=-0.2,
        beta=-0.05,
    )
