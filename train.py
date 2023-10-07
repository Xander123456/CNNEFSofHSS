import time
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model import OneDCNN, TwoDCNN, OneDplusTwoDCNN, OneDplusTwoDCNNAttention
from HeartSoundDataset import HeartSoundDataset

def train(args):
    start = time.time()

    print("Train start...")

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    data_load_time_start = time.time()

    train_dataset = HeartSoundDataset("./dataset", args.train_dataset, length=1)
    val_dataset = HeartSoundDataset("./dataset", args.val_dataset, length=1)
    test_dataset = HeartSoundDataset("./dataset", args.test_dataset, length=0)
    val_dataset_len = len(val_dataset)
#    test_dataset_len = len(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    data_load_time_end = time.time()
    print(f"Dataset loaded, time: {(data_load_time_end - data_load_time_start):.2f}s")
    print(f"Train dataset: {args.train_dataset}, val dataset: {args.val_dataset}, test_dataset: {args.test_dataset}.")

    if args.model_type == "OneDCNN":
        train_OneDCNN(args.epochs, args.batch_size, args.init_lr, train_dataloader, val_dataloader, val_dataset_len, test_dataset, device)
    if args.model_type == "TwoDCNN":
        pass
    if args.model_type == "OneDplusTwoDCNN":
        pass
    if args.model_type == "OneDplusTwoDCNNAttention":
        pass

    end = time.time()
    print(f"Train time: {(end - start):.2f}s")

def train_OneDCNN(epochs, batch_size, lr,train_dataloader, val_dataloader, val_len, test_dataset, device):
    print(f"Train OneDCNN model with {epochs} epochs, {batch_size} batch-size and {lr} learn rate.")

    model = OneDCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_best_acc = 0.93
    vote_val_best_acc = 0.93
    raw_best_acc = 0.93

    for e in range(epochs):
        print(f"--------------------------")
        print(f"Train epoch {e + 1} start.")

        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = F.binary_cross_entropy(pred["output"], y)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (batch + 1) % 100 == 0:
                print(f"Batch {batch + 1}/{len(train_dataloader)} done.")

        print(f"Train epoch {e + 1} end.")

        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                correct += (pred["output"].argmax(1) == y.argmax(1)).type(torch.float).sum().item()


        print(f"Test val accuracy: {(correct / val_len):>0.5f}")
        if correct / val_len > val_best_acc:
            pass

        correct = 0
        with torch.no_grad():
            for i in range(len(test_dataset)):
                X = test_dataset[i][0][None, :].to(device)
                y = test_dataset[i][1][None, :].to(device)

                l = int(X.shape[1] / 32000)
                c = 0
                for j in range(l):

                    x = X[:, j * 32000:(j * 32000 + 32000)]
                    pred = model(x)
                    if pred["output"].argmax(1) == y.argmax(1):
                        c += 1

                if c / l > 0.5:
                    correct += 1

        print(f"Vote val accuracy: {(correct / len(test_dataset)):>0.5f}")
        if correct / val_len > val_best_acc:
            pass

        correct = 0
        with torch.no_grad():
            for i in range(len(test_dataset)):
                x = test_dataset[i][0][None, :].to(device)
                y = test_dataset[i][1][None, :].to(device)
                pred = model(x)
                if pred["output"].argmax(1) == y.argmax(1):
                    correct += 1

        print(f"Raw test accuracy: {(correct / len(test_dataset)):>0.5f}")
        if correct / val_len > val_best_acc:
            pass

            #print(pred["output"].shape)
            #print(pred["embedding"].shape)
