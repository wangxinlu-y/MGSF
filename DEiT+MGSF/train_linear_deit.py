import random
import numpy as np
import os
import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder

# 这里用的是你提供的 Deit 定义
from models import deit_base_patch16_224


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10,
                best_pth='../linear_pth/NWPU20_best_linear_deit.pth',
                last_pth='../linear_pth/NWPU20_last_linear_deit.pth'):
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * train_correct / train_total
        train_avg_loss = train_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            test_acc, test_loss = evaluate(model, test_loader, criterion, device)
            print(f'[Validation] Epoch {epoch + 1}/{num_epochs}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

            if test_acc > best_acc:
                best_acc = test_acc
                os.makedirs(os.path.dirname(best_pth), exist_ok=True)
                torch.save(model.state_dict(), best_pth)

        print(f'[Train] Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

    os.makedirs(os.path.dirname(last_pth), exist_ok=True)
    torch.save(model.state_dict(), last_pth)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total, loss / len(data_loader)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = ImageFolder(root=os.path.join(args.data_path, "train"),
                                transform=data_transform["train"])
    test_dataset = ImageFolder(root=os.path.join(args.data_path, "test"),
                               transform=data_transform["val"])

    print(f"train_image: {len(train_dataset)}")
    print(f"test_image:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print("Building DeiT-BASE backbone...")
    model = deit_base_patch16_224(pretrained=False, num_classes=args.num_classes)
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        print(f"Loading DeiT pretrained weights from: {args.weights}")

        checkpoint = torch.load(args.weights, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        state_dict_filtered = {
            k: v for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(state_dict_filtered)
        msg = model.load_state_dict(model_dict, strict=False)
        print("Load state dict result:", msg)
    else:
        print("No pretrained weights provided, training DeiT from scratch.")
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
        print("Backbone frozen, only head is trainable.")
    else:
        print("All layers are trainable.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr)

    train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        num_epochs=args.epochs,
        best_pth=args.best_pth,
        last_pth=args.last_pth
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")  # 预留，将来你要加 test/eval 也可以
    parser.add_argument('--num_classes', type=int, default=45)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-path', type=str,
                        default=r"/media/magneto/disk/wxl/dataset/NWPU4520_percent")
    parser.add_argument('--weights', type=str,
                        default=r"/media/magneto/disk/wxl/MGSF_TOPK/DEiT/deit_base_patch16_224-b5f2ef4d.pth")
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--best-pth', type=str,
                        default='/media/magneto/disk/wxl/MGSF_TOPK/DEiT/linear_pth/NWPU20_best_linear_deit.pth')
    parser.add_argument('--last-pth', type=str,
                        default='/media/magneto/disk/wxl/MGSF_TOPK/DEiT/linear_pth/NWPU20_last_linear_deit.pth')

    args = parser.parse_args()
    main(args)
