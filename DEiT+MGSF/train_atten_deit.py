import os
import random
import argparse
import numpy as np
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========= 根据你的工程路径修改这两个 import =========
from models import deit_base_patch16_224          # DeiT backbone
from my_dataset import WeightedDataset            # 你自己写的加权数据集
from Dual_atten import ViTWithDualAttention       # SDA / DualAttention 封装



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_image_paths_and_labels(dataset: ImageFolder):
    image_paths = [s[0] for s in dataset.samples]
    image_labels = [s[1] for s in dataset.samples]
    return image_paths, image_labels


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, weights in tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{epochs}]",
        ncols=100
    ):
        images = images.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_per_sample = criterion(outputs, labels)
        loss = (loss_per_sample * weights).mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = val_loss / len(data_loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    assert os.path.exists(args.weight_path), f"weight_path not found: {args.weight_path}"
    weight_data = np.load(args.weight_path)
    train_weights = weight_data["weights"]
    print(f"Loaded sample weights from: {args.weight_path}")
    print(f"  min={train_weights.min():.4f}, max={train_weights.max():.4f}, mean={train_weights.mean():.4f}")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    }

    train_dir = os.path.join(args.data_path, "train")
    test_dir = os.path.join(args.data_path, "test")
    assert os.path.exists(train_dir), f"train dir not found: {train_dir}"
    assert os.path.exists(test_dir), f"test dir not found: {test_dir}"

    base_train_dataset = ImageFolder(root=train_dir, transform=None)
    base_test_dataset = ImageFolder(root=test_dir, transform=None)

    train_paths, train_labels = get_image_paths_and_labels(base_train_dataset)
    test_paths, test_labels = get_image_paths_and_labels(base_test_dataset)

    print(f"Train images: {len(train_paths)}")
    print(f"Test  images: {len(test_paths)}")

    assert len(train_paths) == len(train_weights)

    train_dataset = WeightedDataset(
        weights=train_weights,
        images_path=train_paths,
        images_class=train_labels,
        transform=data_transform["train"]
    )

    test_dataset = WeightedDataset(
        weights=np.ones(len(test_labels)),
        images_path=test_paths,
        images_class=test_labels,
        transform=data_transform["test"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    vit = deit_base_patch16_224(pretrained=False, num_classes=args.num_classes)

    assert os.path.exists(args.linear_pth), f"linear weights not found: {args.linear_pth}"
    print(f"Loading linear DeiT weights from: {args.linear_pth}")
    state_dict = torch.load(args.linear_pth, map_location=device)
    msg = vit.load_state_dict(state_dict, strict=True)
    print("Load state dict result:", msg)

    model = ViTWithDualAttention(vit_model=vit, num_heads=12)
    model.to(device)
    criterion_train = torch.nn.CrossEntropyLoss(reduction="none")
    criterion_val = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    milestone_epoch=150
    scheduler=torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[milestone_epoch],
        gamma=0.5
    )

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.best_pth) or ".", exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_train, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion_val, device
        )
        print(f"[Epoch {epoch+1:03d}/{args.epochs:03d}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"current lr：{current_lr}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.best_pth)
            print(f"  ▶ Best model updated. Saved to: {args.best_pth}")

    torch.save(model.state_dict(), args.last_pth)
    print(f"Training finished. Last model saved to: {args.last_pth}")
    print(f"Best Val Acc: {best_acc:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"/media/magneto/disk/wxl/dataset/NWPU4520_percent"
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        default=r"/media/magneto/disk/wxl/MGSF_TOPK/DEiT/new_weights.npz"
    )
    parser.add_argument(
        "--linear-pth",
        type=str,
        default=r"/media/magneto/disk/wxl/MGSF_TOPK/DEiT/linear_pth/NWPU20_best_linear_deit.pth"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--best-pth",
        type=str,
        default=r"/media/magneto/disk/wxl/MGSF_TOPK/DEiT/NWPU20/best_dual_deit.pth"
    )
    parser.add_argument(
        "--last-pth",
        type=str,
        default=r"/media/magneto/disk/wxl/MGSF_TOPK/DEiT/NWPU20/last_dual_deit.pth"
    )

    args = parser.parse_args()
    main(args)
