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
from deepface import DeepFace
import collections #for ring buffer
import pyttsx3 #TTS; pyttsx3 version 2.6 is needed (newer might not work for windows)

#own python stuff
#from dataconverter import convert_absolute_to_relative, convert_relative_to_class

CONST_BEAUTFUL_ASTERISK = 30 * "*"
CONST_BEAUTIFUL_LINE = 30 * "-"

#this is for speech output
similarity_threshold = 1.0 #if less than this, then you assume it's a match
n_counter_face_detection = 4 #systems needs to detect a learnt person 4 times in a row for successful recognition
n_counter_internal = 0
name_detected_person = 'johndoe' #init name of detected person
system_counter = 0 #number of frame in the script
speech_output_face_recognition = False #init this always to False
speech_output_emotion_detection = False #init this always to False
engine = pyttsx3.init('sapi5') #TTS init
rate = engine.getProperty('rate')   # getting details of current speaking rate
engine.setProperty('rate', 125)     # setting up new voice rate
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)  #changing index, changes voices. 0/1 for female/male

#ring buffer for emotion detection
emotion_ringbuffer = collections.deque(maxlen=5)
emotion_ringbuffer.extend(['emotion1', 'emotion2', 'emotion3', 'emotion4', 'emotion5']) #to change use: emotion_ringbuffer.append('emotion6')
persisting_emotion = 'null'

workers = 0 if os.name == 'nt' else 4

#Determine if an nvidia GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#Define MTCNN module
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=False,
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
#dataset = datasets.ImageFolder(r'images_to_detect')
#dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
#loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


cap = cv.VideoCapture(0)  #0 = read webcam instead of read video file
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    system_counter = system_counter + 1 #increment frame counter

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
    cv.imwrite(r"images_to_detect\unknown_person\frame%d.jpg" %imgcounter, frame) #save as image file
    dataset = datasets.ImageFolder(r'images_to_detect')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

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

            #watch out: this leads to destructible consequences:
            #boxes, probs, points = mtcnn.detect(x, landmarks=True)  # for bounding box (optional)
            # draw bounding box
            #img_draw = x.copy()
            #draw = ImageDraw.Draw(img_draw)
            #for i, (box, point) in enumerate(zip(boxes, points)):
            #    draw.rectangle(box.tolist(), width=5)
            #    for p in point:
            #        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
            #    # extract_face(x, box, save_path='detected_face_{}.png'.format(i))
            #img_draw.save('images_to_detect/unknown_person/annotated_faces.png')
    
    #check if aligned is empty -> no person in frame
    if not aligned:
        print("There is no (known) person in frame")
        # reset counter
        n_counter_internal = 0
        continue
    
    #load known persons
    known_persons_names_path = r'embeddings\names.txt'
    with open(known_persons_names_path, 'rb') as file:
        known_persons = pickle.load(file)

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

    df = pd.DataFrame(dists, columns=unknown_person_name, index=known_persons)
    #print(df)

    #df_relative = df.applymap(convert_absolute_to_relative)
    #print(df_relative)

    #df_message = df_relative.applymap(convert_relative_to_class)
    #print(df_message)

    #another conversion function for converting the relative numbers into similarity values:

    #unique_names = list(dataset.class_to_idx.keys())

    best_match = df.idxmin()
    print("Best match: " + best_match.unknown_person)
    print("\n")

    #identify person if recognitions succeeded several times
    #----------------------------------------------------------------

    name_detected_person_previous = name_detected_person
    name_detected_person = best_match.unknown_person

    #if face is a REAL match AND same face like previous iteration
    if((df.min().values[0] < similarity_threshold) and (name_detected_person_previous == name_detected_person)):
        n_counter_internal = n_counter_internal + 1

    #if face is NOT a REAL match
    else:# (df.min().values[0] >= similarity_threshold) OR different person:
        #reset counter
        n_counter_internal = 0


    #if counter is exceeded
    if(n_counter_internal == n_counter_face_detection): #== not >= otherwise he will tell use several times
        speech_output_face_recognition = True


    #Face Analysis including Emotion Detection
    #**********************************************
    face_analysis = True
    if face_analysis == True:

        #dont do this every frame cause thats wasting a lot of resources
        if((system_counter % 5) == 0): #only do every 5th time etc.

            # detect emotion and other parameters
            img_analysis = DeepFace.analyze(r"images_to_detect\unknown_person\frame%d.jpg" %imgcounter)

            emotion = img_analysis["emotion"]
            age = img_analysis["age"]
            gender = img_analysis["gender"]
            ethnicity = img_analysis["race"]
            dominant_emotion = img_analysis["dominant_emotion"]
            emotion_rounded = {k: round(v, 2) for k, v in emotion.items()}
            ethnicity_rounded = {k: round(v, 2) for k, v in ethnicity.items()}

            #check whether emotion is repetitive: use ringbuffer: if emotion is same several times in a row then use that information
            emotion_ringbuffer.append(dominant_emotion) #add the dominant emotion to ringbuffer
            if(emotion_ringbuffer.__len__() > 0):
                #check for similarity in buffer
                bool_equal = all(elem == emotion_ringbuffer[0] for elem in emotion_ringbuffer) #if all values are the same than the bool is set to true
                if(bool_equal): #equal
                    previous_persisting_emotion = persisting_emotion
                    persisting_emotion = emotion_ringbuffer[0] #get the persisting emotion
                    if(persisting_emotion != previous_persisting_emotion): #not same persisting emotion like last time
                        speech_output_emotion_detection = True #activate speech output
                else: #not equal
                    persisting_emotion = 'null' #dont use


            # print results
            print(CONST_BEAUTIFUL_LINE)
            print("Face Analysis")
            print(CONST_BEAUTIFUL_LINE)

            print("\nEmotion: \n")
            print('\n'.join("{}: {} % ".format(k, v) for k, v in emotion_rounded.items()))
            print("--> Dominant Emotion: " + dominant_emotion)

            print("\nAge: ")
            print(age)

            print("\nGender: ")
            print(gender)

            print("\nEthnicity:  \n")
            print('\n'.join("{}: {} % ".format(k, v) for k, v in ethnicity_rounded.items()))

            print("\n")


    #if requirements is fulfilled then talk
    if (speech_output_face_recognition == True):

        # speech output for person recognition
        print(CONST_BEAUTFUL_ASTERISK)
        speech_output_phrase_face_recognition = "Hey nice to see you, " + best_match.unknown_person + " !"
        print(speech_output_phrase_face_recognition)
        engine.say(speech_output_phrase_face_recognition)
        engine.runAndWait()
        print(CONST_BEAUTFUL_ASTERISK)
        print('\n')
        speech_output_face_recognition = False  # reset

        # speech output for emotion detection


    #if requirements is fulfilled then talk
    if (speech_output_emotion_detection == True):
        #set of emotions: angry, disgust, fear, happy, sad, surprise, neutral
        if(persisting_emotion == 'angry'):
            print(CONST_BEAUTFUL_ASTERISK)
            speech_output_phrase_face_analysis = "Oh, are you angry at me?"
            print(CONST_BEAUTFUL_ASTERISK)

        elif(persisting_emotion == 'disgust'):
            print(CONST_BEAUTFUL_ASTERISK)
            speech_output_phrase_face_analysis = "Hey, what's in your mind?"
            print(CONST_BEAUTFUL_ASTERISK)

        elif(persisting_emotion == 'fear'):
            print(CONST_BEAUTFUL_ASTERISK)
            speech_output_phrase_face_analysis = "Hey, what are you afraid of?"
            print(CONST_BEAUTFUL_ASTERISK)

        elif(persisting_emotion == 'happy'):
            print(CONST_BEAUTFUL_ASTERISK)
            speech_output_phrase_face_analysis = "You seem to be very happy today!"
            print(CONST_BEAUTFUL_ASTERISK)

        elif(persisting_emotion == 'sad'):
            print(CONST_BEAUTFUL_ASTERISK)
            speech_output_phrase_face_analysis = "Hey, what did happen to you?"
            print(CONST_BEAUTFUL_ASTERISK)

        elif(persisting_emotion == 'surprise'):
            print(CONST_BEAUTFUL_ASTERISK)
            speech_output_phrase_face_analysis = "Are you surprised?"
            print(CONST_BEAUTFUL_ASTERISK)

        #else: # (persisting_emotion == 'neutral'):
            #dont do anything

        print(speech_output_phrase_face_analysis)
        engine.say(speech_output_phrase_face_analysis)
        speech_output_emotion_detection = False #reset
        emotion_ringbuffer.extend(['emotion1', 'emotion2', 'emotion3', 'emotion4', 'emotion5']) #reset ringbuffer

    # todo draw bounding boxes,

    imgcounter += 1


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