from __future__ import print_function
from torchvision import datasets, transforms
import albumentations as A
from models import *
import torch.optim as optim
import utils as ut
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from torch_lr_finder import LRFinder


# custom dataset class for albumentations library
class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, train=True):
        self.image_list = image_list
        self.aug = A.Compose({
            A.PadIfNeeded(4, 4),
            A.RandomCrop(32, 32),
            A.Flip(1),
            A.CoarseDropout(1, 8, 8, 1, 8, 8, fill_value=0.473363, mask_fill_value=None),
            A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        })

        self.norm = A.Compose({A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                               })
        self.train = train

    def __len__(self):
        return (len(self.image_list))

    def __getitem__(self, i):
        image, label = self.image_list[i]
        if self.train:
            image = self.aug(image=np.array(image))['image']
        else:
            image = self.norm(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image, dtype=torch.float), label


class DataLoader():
    def __init__(self, batch_size):
        self.cuda = ut.check_for_cuda()
        self.batch_size = batch_size

    def return_loaders(self):
        trainset = datasets.CIFAR10(root='./data', train=True,
                                    download=True)
        testset = datasets.CIFAR10(root='./data', train=False,
                                   download=True)

        train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=True),
                                                   batch_size=self.batch_size,
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train=False),
                                                  batch_size=self.batch_size,
                                                  shuffle=False, num_workers=1)
        return train_loader, test_loader, trainset


class Train():
    def __init__(self):
        self.train_losses = []
        self.train_acc = []

    def train_model(self, model, device, train_loader, optimizer, criterion, scheduler):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        lr_trend = []
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)
            self.train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # updating LR
            if scheduler:
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id = {batch_idx} Accuracy = {100 * correct / processed:0.2f}')
            self.train_acc.append(100 * correct / processed)
        return sum(self.train_losses) / len(self.train_losses), sum(self.train_acc) / len(self.train_acc), lr_trend


class Test():
    def __init__(self):
        self.test_losses = []
        self.test_acc = []

    def test_model(self, model, device, test_loader, criterion):
        model.eval()
        pred_epoch = []
        target_epoch = []
        data_epoch = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                if (len(pred_epoch) < 10):
                    b_target = target.cpu().numpy()
                    b_pred = pred.view_as(target).cpu().numpy()
                    for i in range(len(b_target)):
                        if (len(pred_epoch) < 10):
                            if (b_target[i] != b_pred[i]):
                                pred_epoch.append(b_pred[i])
                                target_epoch.append(b_target[i])
                                data_epoch.append(data[i].cpu().numpy())
                self.test_losses.append(test_loss.item())
                self.test_acc.append(100. * (correct / len(data)))
        test_loss = sum(self.test_losses) / len(test_loader.dataset)

        print("\n Test set: Avergae loss: {:4f}, Accuracy = {}/{}({:.2f}%)\n".format(test_loss, sum(self.test_acc),
                                                                                     len(test_loader.dataset),
                                                                                     (sum(self.test_acc) / len(
                                                                                         self.test_acc))))
        return test_loss, sum(self.test_acc) / len(self.test_acc), pred_epoch, target_epoch, data_epoch


def train_and_test_model(model, train_loader, test_loader, device, criterion, optimizer, epochs, scheduler):
    ut.print_model_summary(model)
    train = Train()
    test = Test()
    train_losses_all_epochs = []
    train_acc_all_epochs = []
    test_losses_all_epochs = []
    test_acc_all_epochs = []

    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train_loss, train_acc, lr_trend = train.train_model(model, device, train_loader, optimizer, criterion, scheduler)
        train_losses_all_epochs.append(train_loss)
        train_acc_all_epochs.append(train_acc)
        test_loss, test_acc, pred, target, data = test.test_model(model, device, test_loader, criterion)
        test_losses_all_epochs.append(test_loss)
        test_acc_all_epochs.append(test_acc)
        #print(train_losses_all_epochs, train_acc_all_epochs, test_losses_all_epochs, test_acc_all_epochs)
    return train_losses_all_epochs, train_acc_all_epochs, test_losses_all_epochs, test_acc_all_epochs, pred, target, data, lr_trend


def max_lr_finder(batch_size, device, trainset, model, optimizer, criterion):
    train_loader_noaug = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train=False),
                                                     batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader_noaug, start_lr=0.0001, end_lr=1, num_iter=200)
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    ler_rate = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("Max LR is {}".format(ler_rate))
    lr_finder.reset()
    return ler_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--loss")
    parser.add_argument("--ass", type=int)
    args = parser.parse_args()

    train_loader, test_loader, trainset = DataLoader(args.batch_size).return_loaders()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.ass == 8:
        model = CustomResNet().to(device)
    elif args.ass == 9:
        model = Transformer().to(device)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if args.loss == 'cross_ent':
        criterion = nn.CrossEntropyLoss()
    else:
        if args.loss == "nll":
            criterion = nn.NLLLoss()
    max_lr = max_lr_finder(args.batch_size, device, trainset, model, optimizer, criterion)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=max_lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs,
                                                    pct_start=0.208,
                                                    div_factor=10,
                                                    three_phase=False,
                                                    final_div_factor=100,
                                                    anneal_strategy='linear'
                                                    )
    train_losses, train_acc, test_losses, test_acc, pred, target, data, lr_trend = train_and_test_model(model, train_loader,
                                                                                              test_loader, device,
                                                                                              criterion,
                                                                                              optimizer,
                                                                                              args.epochs,
                                                                                              scheduler)

    #ut.draw_train_test_acc_loss(train_losses, train_acc, test_losses, test_acc)
    #ut.draw_misclassified_images(pred, target, data, "misclassified with resnet")
    #ut.draw_gradcam_images(model, data, pred, target, device)


if __name__ == "__main__":
    main()
