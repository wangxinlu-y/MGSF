import random
import os
import torch
import argparse
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_model import vit_base_patch16_224_in21k, vit_large_patch16_224_in21k
from my_dataset import WeightedDataset
from Dual_atten import ViTWithDualAttention

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def get_image_paths_and_labels(dataset):
    return [s[0] for s in dataset.samples], [s[1] for s in dataset.samples]


# 在 train_dual_model 函数中添加动态选择统计的监控
def train_dual_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    best_acc = 0.0
    best_epoch = 0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    # 添加动态选择统计记录
    token_ratios, dim_ratios = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels, weights in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_per_sample = criterion(outputs, labels)
            weighted_loss = (loss_per_sample * weights).mean()
            weighted_loss.backward()
            optimizer.step()

            train_loss += weighted_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



        train_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        current_test_acc, current_test_loss = validate(model, test_loader, criterion, device)
        test_accs.append(current_test_acc)
        test_losses.append(current_test_loss)

        if current_test_acc > best_acc:
            best_acc = current_test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'CUC50/01_best_dual_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Test Loss: {current_test_loss:.4f} Acc: {current_test_acc:.2f}%')
        print(f"\n[Epoch {epoch + 1}] Learned dynamic ratios:")
        for i, mod in enumerate(model.dual_modules):
            stat = mod.get_dynamic_stats()
            print(f"  Block {i}: token={stat['token_ratio']:.4f}, dim={stat['dim_ratio']:.4f}")

    print(f'\nbestepoch: {best_epoch}, OA: {best_acc:.2f}%')
    np.savez('training_metrics.npz',
             train_losses=train_losses, train_accs=train_accs,
             test_losses=test_losses, test_accs=test_accs,
             token_ratios=token_ratios, dim_ratios=dim_ratios)
    torch.save(model.state_dict(), 'CUC50/01_last_dual_model.pth')

def validate(model, test_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels).mean()
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, val_loss / len(test_loader)

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    weight_data = np.load(args.weight_path)
    train_weights = weight_data['weights']

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dir = os.path.join(args.data_path, "train")
    test_dir = os.path.join(args.data_path, "test")
    train_data = ImageFolder(root=train_dir, transform=data_transform["train"])
    test_data = ImageFolder(root=test_dir, transform=data_transform["test"])
    train_paths, train_labels = get_image_paths_and_labels(train_data)
    test_paths, test_labels = get_image_paths_and_labels(test_data)
    assert len(train_paths) == len(train_weights)

    train_dataset = WeightedDataset(weights=train_weights, images_path=train_paths,
                                    images_class=train_labels, transform=data_transform["train"])
    test_dataset = WeightedDataset(weights=np.ones(len(test_labels)), images_path=test_paths,
                                   images_class=test_labels, transform=data_transform["test"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.vit_type == 'base':
        vit = vit_base_patch16_224_in21k(num_classes=args.num_classes)
    elif args.vit_type == 'large':
        vit = vit_large_patch16_224_in21k(num_classes=args.num_classes)
    else:
        raise ValueError("Unsupported vit-type")

    if args.weights:
        print(f"Loading pretrained weights from {args.weights}")
        pretrained_dict = torch.load(args.weights, map_location=device)
        model_dict = vit.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        vit.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(filtered_dict)}/{len(pretrained_dict)} parameters")

    model = ViTWithDualAttention(vit_model=vit, num_heads=vit.blocks[0].attn.num_heads)  # ✅ 动态获取 num_heads
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    train_dual_model(model, train_loader, test_loader, criterion, optimizer, device, args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-path', type=str, default=r"/media/magneto/disk/wxl/dataset/CUC50_percent")
    parser.add_argument('--weight-path', type=str, default='new_weights.npz')
    parser.add_argument('--weights', type=str, default='linear_pth/CUC50_best_linear.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--vit-type', type=str, default='base', choices=['base', 'large'])  # ✅ 新增参数
    args = parser.parse_args()
    main(args)