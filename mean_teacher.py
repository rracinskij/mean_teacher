#mean teacher code based on https://github.com/CuriousAI/mean-teacher/tree/master/pytorch

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument("--cons_weight", default=10, help="consistency weight")
args = parser.parse_args()


#https://github.com/pytorch/examples/blob/master/mnist/main.py (Pytorch MNIST example net)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

train_transformation = TransformTwice(transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]))

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

def train_mt(model, ema_model, train_loader, optimizer, epoch, step_counter):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=-1).cuda()
    consistency_criterion = softmax_mse_loss
    residual_logit_criterion = symmetric_mse_loss
    alpha = 0.999
    consistency_weight = int(args.cons_weight)
    model.train()
    ema_model.train()
    epoch_loss = 0

    for i, ((input,ema_input), target) in enumerate(train_loader):

        input_var = input.cuda()
        with torch.no_grad():
            ema_input_var = ema_input.cuda()
        target_var = target.cuda()
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(-1).sum()

        model_out = model(input_var)
        ema_model_out = ema_model(ema_input_var)


        logit1 = model_out
        ema_logit = ema_model_out

        ema_logit = ema_logit.detach().data

        class_logit, cons_logit = logit1, logit1
        res_loss = 0

        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
        consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
        loss = class_loss + consistency_loss + res_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_counter += 1
        epoch_loss += loss
        update_ema_variables(model, ema_model, alpha, step_counter)

    print("epoch: {} , epoch_loss: {:.3f}".format(epoch, epoch_loss))
    print("loss, class_loss, consistency_loss: ", loss.cpu().data.numpy(), class_loss.cpu().data.numpy(), consistency_loss.cpu().data.numpy())
    return step_counter

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    step_counter = 0
    model = Net().cuda()
    ema_model = Net().cuda()
    for param in ema_model.parameters():
        param.detach_()

    num_indices = 1000

    indices = list(range(num_indices*10)) #take x10 images
    train_dataset_less_labels = MNIST('./data', train=True, download=True, transform = train_transformation)
    train_dataset_less_labels.train_labels[num_indices:] = -1 #set labels from given index to -1 (=no label)
    train_dataset_less_labels = Subset(train_dataset_less_labels, indices)
    train_loader_less_labels = torch.utils.data.DataLoader(train_dataset_less_labels, batch_size=64, shuffle=True)
    image, label = next(iter(train_loader_less_labels))
    print("sample labels: ", label) #check the labels

    test_dataset = MNIST('./data', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.05)

    for epoch in range(500):
        print("epoch: ", epoch)
        train_mt(model, ema_model, train_loader_less_labels, optimizer, epoch, step_counter)
        test(model, test_loader)


if __name__ == '__main__':
    main()
