import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

root_dir = os.path.dirname(__file__)

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)


class NeuraNetwork(nn.Module):
    def __init__(self):
        super(NeuraNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 
                      'cpu')
model = NeuraNetwork()
model.to(device=device)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch,(X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

def start_train(epoch, batch_size, lr):
    train_dataloader = DataLoader(training_data, batch_size,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size,shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for t in range(epoch):
        print(f'Epoch {t+1}\n' + '-' * 50)
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print('Training is finish.')

    torch.save(model.state_dict(), os.path.join(root_dir,'model.pth'))
    

def start_test(model_name):
    model = NeuraNetwork()
    model.load_state_dict(torch.load(model_name))
    idx = random.randint(0, len(test_data)-1)    
    image,label = test_data[idx]
    print(f'test idx:{idx} label is {label}')
    pred = model(image)
    m = pred.argmax(1).item()
    print(f'prediction label: {m}')

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    plt.pause(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='model.pth')
    args = parser.parse_args()

    if args.type == 'train':
        print('start trainning...')
        epoch = (args.epoch if args.epoch > 0 and args.epoch < 1000 else 10)
        bs = (args.bs if args.bs > 0 and args.bs < 128 else 64)
        lr = (args.lr if args.lr > 0 and args.lr < 0.1 else 1e-3)
        start_train(epoch, bs, lr)
    elif args.type == 'board':
        train_dataloader = DataLoader(training_data, 8,shuffle=True)
        images, labels = next(iter(train_dataloader))
        grid = torchvision.utils.make_grid(images)
        writer = SummaryWriter()
        writer.add_image('image',grid,0)
        writer.close()
    else:
        print('test module')
        model_name = args.model if os.path.exists(os.path.join(root_dir,args.model)) else os.path.join(root_dir, 'model.pth')
        start_test(model_name)