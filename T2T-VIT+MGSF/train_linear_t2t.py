import random
import numpy as np
import os
import torch
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

from T2T.t2t_vit import t2t_vit_14


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model(model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                device,
                num_epochs: int = 100,
                best_pth: str = r'linear_pth\CUC50_best_linear_t2t.pth',
                last_pth: str = r'linear_pth\CUC50_last_linear_t2t.pth'):
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader,
                                   desc=f'Epoch {epoch + 1}/{num_epochs}'):
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

    train_dataset = ImageFolder(
        root=os.path.join(args.data_path, "train"),
        transform=data_transform["train"]
    )
    test_dataset = ImageFolder(
        root=os.path.join(args.data_path, "test"),
        transform=data_transform["val"]
    )

    print(f"train_image: {len(train_dataset)}")
    print(f"test_image:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print("Building T2T-ViT-14 backbone...")
    model = t2t_vit_14(pretrained=False, img_size=224, num_classes=args.num_classes)
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        print(f"Loading T2T-ViT-14 pretrained weights from: {args.weights}")

        checkpoint = torch.load(args.weights, map_location=device)
        print("Checkpoint type:", type(checkpoint))

        if isinstance(checkpoint, dict):
            if "state_dict_ema" in checkpoint:
                print("Using state_dict_ema from checkpoint")
                state_dict = checkpoint["state_dict_ema"]
            elif "state_dict" in checkpoint:
                print("Using state_dict from checkpoint")
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                print("Using model from checkpoint")
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        stripped_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[len("module."):] if k.startswith("module.") else k
            stripped_state_dict[new_k] = v

        model_state = model.state_dict()
        filtered_state_dict = {}
        for k, v in stripped_state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state_dict[k] = v

        print(f"Shape-matched params: {len(filtered_state_dict)} / {len(model_state)}")
        model_state.update(filtered_state_dict)
        msg = model.load_state_dict(model_state, strict=True)
        print("Load state dict result: OK")
        not_loaded = [k for k in model.state_dict().keys() if k not in filtered_state_dict]
        print("Not loaded or randomly initialized keys (first 20):")
        for k in not_loaded[:20]:
            print("  ", k)
    else:
        print("No pretrained weights provided, training T2T-ViT from scratch.")
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
        print("Backbone frozen, only head is trainable.")
    else:
        print("All layers are trainable.")

    total_params = 0
    trainable_params = 0
    print("Trainable parameters (requires_grad=True):")
    for name, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
            print(f"  {name}: {tuple(p.shape)}")

    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

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
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--data-path', type=str,
                        default=r"/media/magneto/disk/wxl/dataset/CUC50_percent")
    parser.add_argument('--weights', type=str,
                        default=r"/media/magneto/disk/wxl/MGSF_TOPK/T2T-VIT/81.5_T2T_ViT_14.pth")

    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--best-pth', type=str,
                        default=r"/media/magneto/disk/wxl/MGSF_TOPK/T2T-VIT/linear_pth/CUC50_best_linear_t2t.pth")
    parser.add_argument('--last-pth', type=str,
                        default=r"/media/magneto/disk/wxl/MGSF_TOPK/T2T-VIT/linear_pth/CUC50_last_linear_t2t.pth")
    args = parser.parse_args()
    main(args)
