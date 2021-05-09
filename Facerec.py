"""
Face detection and recognition
Based on facenet_pytorch python package, MTCNN
Inception Resnet V1 pretrained on the VGGFace2 dataset
This project is developed by Kai Mueller, Fabian Luettel and Pauline Weimann
"""

#Import packages
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle
import cv2 as cv
import time

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
#dataset = datasets.ImageFolder(r'images_to_detect')
#dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
#loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


cap = cv.VideoCapture(0)  #0 = read webcam instead of read video file
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:

    time.sleep(1) #maybe deactiviate depending on requirements

    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
    imgcounter = 0
    cv.imwrite(r"images_to_detect\unknown_person\frame%d.jpg" %imgcounter, frame)
    dataset = datasets.ImageFolder(r'images_to_detect')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    imgcounter += 1

    #Perfom MTCNN facial detection
    #Iterate through the DataLoader object and detect faces and associated detection probabilities for each.
    #The MTCNN forward method returns images cropped to the detected face, if a face was detected.
    #By default only a single detected face is returned - to have MTCNN return all detected faces, set keep_all=True when creating the MTCNN object above.
    #To obtain bounding boxes rather than cropped face images, you can instead call the lower-level mtcnn.detect() function.
    aligned = []
    unknown_person_name = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            unknown_person_name.append(dataset.idx_to_class[y])
    
    #check if aligned is empty -> no person in frame
    if not aligned:
        print("There is no (known) person in frame")
        continue
    
    #load known persons
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
    #print(df)

    #df_relative = df.applymap(convert_absolute_to_relative)
    #print(df_relative)

    #df_message = df_relative.applymap(convert_relative_to_class)
    #print(df_message)

    #another conversion function for converting the relative numbers into similarity values:

    #unique_names = list(dataset.class_to_idx.keys())

    best_match = df.idxmin()
    print("\n---------- Best match:  \n")
    print(best_match)

    # todo draw bounding boxes,
    # todo workers etc.

        
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
'''