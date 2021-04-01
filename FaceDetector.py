# https://github.com/timesler/facenet-pytorch
# https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb


# Packages
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

# Determine if an nvidia GPU is available¶
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN module
mtcnn = MTCNN(keep_all=True, device=device)

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN() #mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open("input/image_example.jpg")

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path="output/cropped_image.jpg")

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))



# Get a sample video
video = mmcv.VideoReader('input/example_video_short.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

display.Video('input/example_video_short.mp4', width=640)

# Run video through MTCNN
frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')


# Display detections
# d = display.display(frames_tracked[0], display_id=True)
# i = 1
# try:
#     while True:
#         d.update(frames_tracked[i % len(frames_tracked)])
#         i += 1
# except KeyboardInterrupt:
#     pass


# Save tracked video
dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_tracked = cv2.VideoWriter('output/example_video_short_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()