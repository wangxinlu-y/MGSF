import os
import torch
import numpy as np
import math
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from vit_model import vit_base_patch16_224_in21k, vit_large_patch16_224_in21k


class WeightCalculator:
    def __init__(self, model_path, data_path, num_classes=45, device='cuda:0', vit_type='base'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.data_path = data_path

        if vit_type == 'base':
            self.model = vit_base_patch16_224_in21k(num_classes=num_classes)
        elif vit_type == 'large':
            self.model = vit_large_patch16_224_in21k(num_classes=num_classes)
        else:
            raise ValueError("Unsupported vit_type")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 定义transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def calculate_weights(self, T1=1.5, T2=1.5, alpha=-0.05, beta=-0.05):
        train_dir = os.path.join(self.data_path, "train")
        dataset = ImageFolder(root=train_dir, transform=self.transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

        weights = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                all_probabilities.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # 计算熵
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
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

    def save_weights(self, save_path='new_weights.npz', **kwargs):
        weights, labels, probs = self.calculate_weights(**kwargs)
        np.savez(save_path,
                 weights=weights,
                 labels=labels,
                 probabilities=probs)
        print(f"Weights saved to {save_path}")
        print(f"Weight stats - Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")


if __name__ == '__main__':
    calculator = WeightCalculator(
        model_path='linear_pth/CUC50_best_linear.pth',
        data_path=r"/media/magneto/disk/wxl/dataset/CUC50_percent",
        num_classes=14,
        device='cuda:0',
        vit_type = 'base'
    )
    calculator.save_weights(
        T1=2.2,
        T2=1.65,
        alpha=-0.3,
        beta=-0.05,
    )
