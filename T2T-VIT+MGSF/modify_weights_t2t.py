import os
import math
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 用你项目里的 T2T-ViT-14
from T2T.t2t_vit import t2t_vit_14


class WeightCalculatorT2T:
    def __init__(self,
                 model_path,
                 data_path,
                 num_classes=21,
                 device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.data_path = data_path

        print("Building T2T-ViT-14 for weight calculation...")
        self.model = t2t_vit_14(pretrained=False, img_size=224, num_classes=num_classes)
        assert os.path.exists(model_path), f"model_path 不存在: {model_path}"
        print(f"Loading linear probe weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

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
                            batch_size=32,
                            shuffle=False,
                            drop_last=False)

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
                     save_path='D:\MGSF_TOPK\T2T-VIT\T2T\new_weights_t2t.npz',
                     **kwargs):
        weights, labels, probs = self.calculate_weights(**kwargs)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        np.savez(save_path,
                 weights=weights,
                 labels=labels,
                 probabilities=probs)

        print(f"save: {save_path}")
        print(f"weight: min={weights.min():.4f}, "
              f"max={weights.max():.4f}, "
              f"mean={weights.mean():.4f}")


if __name__ == '__main__':
    calculator = WeightCalculatorT2T(
        model_path=r"/media/magneto/disk/wxl/MGSF_TOPK/T2T-VIT/linear_pth/CUC50_best_linear_t2t.pth",
        data_path=r"/media/magneto/disk/wxl/dataset/CUC50_percent",
        num_classes=14,
        device='cuda:0'
    )

    calculator.save_weights(
        save_path='new_weights_t2t.npz',
        T1=2.0,
        T2=1.5,
        alpha=-0.2,
        beta=-0.05,
    )
