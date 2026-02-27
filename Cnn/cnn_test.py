from PIL import Image
import torchvision.transforms as transforms
from cnn_arch import SimpleCNN

model = SimpleCNN()
# Load image
image = Image.open("cat.png").convert("RGB")

# Transform image
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # Resize to match model
    transforms.ToTensor()          # Convert to tensor
])

input_tensor = transform(image)

# Add batch dimension (VERY IMPORTANT)
input_tensor = input_tensor.unsqueeze(0)

# Forward pass
output = model(input_tensor)

print("Output shape:", output.shape)
print("Raw output:", output)