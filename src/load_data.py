import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Basic transform: just convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# 2. Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 3. Print basic info
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
print("Classes:", train_dataset.classes)

# 4. Visualize 5 sample images
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i in range(5):
    image, label = train_dataset[i]
    axes[i].imshow(image.permute(1, 2, 0))
    axes[i].set_title(train_dataset.classes[label])
    axes[i].axis("off")

plt.tight_layout()
plt.show()
