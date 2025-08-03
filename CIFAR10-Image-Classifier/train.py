import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# helper function to show an image
def imgshow(img, mean, std):
    """
    Unnormalize the image and convert it to numpy array
    Args:
        img: image to show
        mean: mean of the dataset
        std: standard deviation of the dataset
    """
    # unnormalize the image - reshape mean and std for broadcasting
    img = img * std.view(3, 1, 1) + mean.view(3, 1, 1)

    # clamp the image to 0-1
    img = torch.clamp(img, 0, 1)

    # move to CPU and convert to numpy array
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get device helper function
def get_device():
    """
    Get the device to use for training and inference
    """
    # Check for CUDA availability
    if torch.cuda.is_available():
        torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def calculate_stats(device: torch.device):
    # dataset 
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # Set to 0 for macOS

    # Calculate mean and std
    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    nb_samples = 0

    # Calculate mean and std
    for data, _ in loader:
        data = data.to(device) # move tensor to gpu/nlu device
        batch_samples = data.size(0) # number of samples in the batch
        data = data.view(batch_samples, data.size(1), -1) # flatten the data
        mean += data.mean(2).sum(0) # sum of all the pixels in the batch
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean.cpu(), std.cpu()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2) # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16 * 5 * 5 input features, 120 output features
        self.fc2 = nn.Linear(120, 84) # 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 10) # 84 input features, 10 output features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_testset(transform: transforms.Compose):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    return testloader

def get_trainset(transform: transforms.Compose):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    return trainloader

if __name__ == '__main__':
    # load device
    device = get_device() 
    print(f"Using device: {device}")

    # mean, std = calculate_stats(device)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device)
    print(f"Mean: {mean}, Std: {std}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainloader = get_trainset(transform)
    testloader = get_testset(transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get a batch of training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # move images to the device
    images = images.to(device)
    labels = labels.to(device)

    # make a grid of images
    img_grid = torchvision.utils.make_grid(images)

    # print labels
    print('GroundTruth: ' + ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    imgshow(img_grid, mean.to(device), std.to(device))

    # load model
    net = Net().to(device)
    net.load_state_dict(torch.load('./models/cifar_net.pth'))
    net.eval()

    # get predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # print predictions
    print('Predicted: ' + ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Prediction accuracy in tensor device
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    # Prediction accuracy in cpu

    # # get a batch of training images
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)

    # # move images to the device
    # images = images.to(device)
    # labels = labels.to(device)

    # # move mean and std to the device
    # mean = mean.to(device)
    # std = std.to(device)

    # # make a grid of images
    # img_grid = torchvision.utils.make_grid(images)
    # imgshow(img_grid, mean, std)

    # net = Net().to(device)

    # # define a loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # # train the network
    # for epoch in range(4):
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs
    #         inputs, labels = data
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999: # print every 2000 mini-batches
    #             print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
    #             running_loss = 0.0

    # print("Finished Training")

    # # save the model
    # PATH = './models/cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

    # print("Saved model to ", PATH)
