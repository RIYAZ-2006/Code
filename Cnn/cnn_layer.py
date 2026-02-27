import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from cnn_arch import SimpleCNN

model = SimpleCNN()
# Load image
image = Image.open("cat.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Pass through first layer only
with torch.no_grad():
    layer1_output = model.conv1(input_tensor)
    layer1_output = torch.relu(layer1_output)

# Plot first layer feature maps
fig, axes = plt.subplots(4, 2, figsize=(8,8))
for i, ax in enumerate(axes.flat):
    ax.imshow(layer1_output[0, i].cpu(), cmap='gray')
    ax.axis('off')

plt.suptitle("First Layer Feature Maps (Edges)")
plt.show()