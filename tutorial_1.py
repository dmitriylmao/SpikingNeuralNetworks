import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML


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
num_steps = 100
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

print(spike_data.size())

spike_data_sample = spike_data[:, 0, 0]
print(spike_data_sample.size()) 

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
anim.save("spike_mnist_test.gif")

spike_data = spikegen.rate(data_it, num_steps=num_steps, gain=0.25)

spike_data_sample2 = spike_data[:, 0, 0]
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample2, fig, ax)
anim.save("spike_mnist_test_gain025.gif")
print(f"The corresponding target is: {targets_it[0]}")


plt.figure(facecolor="w")

# Без изменения (частота 100%)
plt.subplot(1,2,1)
plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 1')

# С пониженной частотой (25%)
plt.subplot(1,2,2)
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 0.25')

plt.show()

spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))

# raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample2, ax, s=1.5, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()


idx = 210  # index into 210th neuron

fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

splt.raster(spike_data_sample.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.show()


def convert_to_time(data, tau=5, threshold=0.01):
  spike_time = tau * torch.log(data / (data - threshold))
  return spike_time

raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
spike_times = convert_to_time(raw_input)

plt.plot(raw_input, spike_times)
plt.xlabel('Input Value')
plt.ylabel('Spike Time (s)')
plt.show()

spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)