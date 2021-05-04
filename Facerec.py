"""
Face detection and recognition inference pipeline

The following example illustrates how to use the facenet_pytorch python package to perform face detection and recogition on an image dataset using an Inception Resnet V1 pretrained on the VGGFace2 dataset.

The following Pytorch methods are included:

    Datasets
    Dataloaders
    GPU/CPU processing
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle

#own python stuff
#from dataconverter import convert_absolute_to_relative, convert_relative_to_class

#pick mode here
mode = "livevideo"
#mode = "singleimages"

workers = 0 if os.name == 'nt' else 4

#Determine if an nvidia GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#Define MTCNN module
#Default params shown for illustration, but not needed. Note that, since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.
#See help(MTCNN) for more details.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

#checkpoint = torch.load(r'save\model.pt')

#Define Inception Resnet V1 module
#Set classify=True for pretrained classifier. For this example, we will use the model to output embeddings/CNN features. Note that for inference, it is important to set the model to eval mode.
#See help(InceptionResnetV1) for more details.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#resnet.load_state_dict(checkpoint['model_state_dict'])
#resnet.eval()
#num_classes = checkpoint['num_classes']

#Define a dataset and data loader
#We add the idx_to_class attribute to the dataset to enable easy recoding of label indices to identity names later one.
def collate_fn(x):
    return x[0]

#Load test data
#if mode == "singleimages":
dataset = datasets.ImageFolder(r'images_to_detect')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


'''
elif mode == "livevideo"
    cap = cv.VideoCapture(0)  #0 = read webcam
    counter = 0
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        counter = counter + 1;
        if ((counter%200)==0):
            print(counter)
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break

            #face detection here for each frame:
            x_aligned, prob = mtcnn(frame, return_prob=True) #todo is frame das richtige argument?
            if x_aligned is not None:
                print('Face detected with probability: {:8f}'.format(prob))

            # todo draw bounding boxes,
            # todo workers etc.

            # detect faces
            boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

            # visualize
            # plt.subplot() is a function that returns a tuple containing a figure and axes objects
            # use fig to change figure-level attributes or save figure as an image file later (fig.savefig('filename.png')
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.imshow(frame)
            ax.axis('off')

            for box, landmark in zip(boxes, landmarks):
                ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
                ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
            fig.show()
            
    cap.release()
    cv.destroyAllWindows()
            '''

#Perfom MTCNN facial detection
#Iterate through the DataLoader object and detect faces and associated detection probabilities for each. The MTCNN forward method returns images cropped to the detected face, if a face was detected. By default only a single detected face is returned - to have MTCNN return all detected faces, set keep_all=True when creating the MTCNN object above.
#To obtain bounding boxes rather than cropped face images, you can instead call the lower-level mtcnn.detect() function. See help(mtcnn.detect) for details.
aligned = []
unknown_person_name = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        unknown_person_name.append(dataset.idx_to_class[y])

know_persons_names_path = r'embeddings\names.txt'
with open(know_persons_names_path, 'rb') as file:
    know_persons = pickle.load(file)
#Calculate image embeddings
#MTCNN will return images of faces all the same size,
# enabling easy batch processing with the Resnet
# recognition module. Here, since we only have a few
# images, we build a single batch and perform inference on it.
#For real datasets, code should be modified to control batch sizes
# being passed to the Resnet, particularly if being processed on a GPU.
# For repeated testing, it is best to separate face detection (using MTCNN)
# from embedding or classification (using InceptionResnetV1), as calculation of cropped faces or bounding boxes
# can then be performed a single time and detected faces saved for future use.
aligned = torch.stack(aligned).to(device)
unknown_embedding = resnet(aligned).detach().cpu()
learned_embeddings = torch.load('embeddings\embeddings.pt')

#Print distance matrix for classes
dists = [(element - unknown_embedding).norm().item() for element in learned_embeddings]

#dists = [[(e1 - e2).norm().item() for e2 in embedding] for e1 in embedding]
#Debugger:
#dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
#dists = np.array(dists)
#dists
#formt ein numpyarray der Ergebnisse

df = pd.DataFrame(dists, columns=unknown_person_name, index=know_persons)
print(df)

#df_relative = df.applymap(convert_absolute_to_relative)
#print(df_relative)

#df_message = df_relative.applymap(convert_relative_to_class)
#print(df_message)

#another conversion function for converting the relative numbers into similarity values:

#unique_names = list(dataset.class_to_idx.keys())

best_match = df.idxmin()
print("\n---------- Best match:  \n")
print(best_match)