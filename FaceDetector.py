# https://github.com/timesler/facenet-pytorch
# https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb
from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN() #mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

img = Image.open("input/image_example.jpg")

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path="output/cropped_image.jpg")

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))