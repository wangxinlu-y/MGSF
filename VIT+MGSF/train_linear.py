import random
import numpy as np
import os
import torch
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_model import vit_base_patch16_224_in21k, vit_large_patch16_224_in21k
from torchvision.datasets import ImageFolder

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    best_acc = 0.0
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
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

        train_accuracy = 100 * train_correct / train_total
        train_avg_loss = train_loss / len(train_loader)

        if(epoch+1)%10==0 or (epoch+1)==num_epochs:
            test_acc,test_loss=evaluate(model,test_loader,criterion,device)
            print(f'[Validation] Epoch{epoch+1}/{num_epochs},Test Loss:{test_loss:.4f},Test Acc:{test_acc:.2f}%')
            if test_acc > best_acc:
              best_acc = test_acc
              torch.save(model.state_dict(), 'linear_pth/CUC50_best_linear.pth')

        print(f'[Train]Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
    torch.save(model.state_dict(), 'linear_pth/CUC50_last_linear.pth')

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
    return 100 * correct / total, loss / len(data_loader)

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = ImageFolder(root=os.path.join(args.data_path, "train"), transform=data_transform["train"])
    test_dataset = ImageFolder(root=os.path.join(args.data_path, "test"), transform=data_transform["val"])
    print(f"train_image:{len(train_dataset)}")
    print(f"test_image:{len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.vit_type == 'base':
        model = vit_base_patch16_224_in21k(num_classes=args.num_classes)
        print("Using ViT-BASE backbone")
    elif args.vit_type == 'large':
        model = vit_large_patch16_224_in21k(num_classes=args.num_classes)
        print("Using ViT-LARGE backbone")
    else:
        raise ValueError("Unsupported vit-type")

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        print(f"Loading pretrained weights from: {args.weights}")
        pretrained_dict = torch.load(args.weights, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print("No pretrained weights provided, training from scratch.")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-path', type=str, default=r"/media/magneto/disk/wxl/dataset/CUC50_percent")
    parser.add_argument('--weights', type=str, default=r'/media/magneto/disk/wxl/Vit_me_00_large/VIT_pretrained_model/vit_base_patch16_224_in21k.pth')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--vit-type', type=str, default='base', choices=['base', 'large'])
    args = parser.parse_args()
    main(args)
