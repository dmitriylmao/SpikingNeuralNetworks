import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
from snntorch import spikegen

batch_size = 128 #это партия данных
data_path = './data/mnist'  # путь куда будут скачиваться данные
num_classes = 10  # 10 цифр от 0 до 9
dtype = torch.float

# Преобразования: 28x28, grayscale, тензор, нормализация от 0 до 1
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# Загрузка тренировочного датасета
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

subset = 10  # уменьшаем в 10 раз (6000 вместо 60000)
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

#итератор, который выдаёт батчи по 128 изображений
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

dataiter = iter(train_loader)
images, labels = next(dataiter)

# Iterate through minibatches
num_steps = 10
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

print(spike_data.size())