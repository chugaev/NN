import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from Train import Net

def test_nn(batch_size=200):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    net = Net()
    net.load_state_dict(torch.load("my_net"), strict=False)

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    torch.no_grad()
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        test_loss += criterion(net_out, target).item()
        pred = net_out.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    test_nn()