"""
Face detection and recognition
Based on facenet_pytorch python package, MTCNN
Inception Resnet V1 pretrained on the VGGFace2 dataset
This project is developed by Kai Mueller, Fabian Luettel and Pauline Weimann
"""

'''
Luettel, Fabian
Mueller, Kai
Weimann, Pauline
2021
'''

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
import threading #asynchronous stuff

#own python stuff
#from dataconverter import convert_absolute_to_relative, convert_relative_to_class

CONST_BEAUTFUL_ASTERISK = 30 * "*"
CONST_BEAUTIFUL_LINE = 30 * "-"

similarity_threshold = 1.0 #if less than this, then you assume it's a match
delta_first_secon_bestmatch = 0.05 #gap between best and second best match
frequence_face_analysis = 8 #activate face analysis on every n_th frame

#this is for speech output
n_counter_face_detection = 3 #systems needs to detect a learnt person n times in a row for successful recognition
name_detected_person = 'johndoe' #init name of detected person
name_detected_person_primary = 'johndoe' #init name of detected person
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
    post_process=True,
    keep_all=True,
    device=device
)

#Define Inception Resnet V1 module
#Set classify=True for pretrained classifier. For this example, we will use the model to output embeddings/CNN features. Note that for inference, it is important to set the model to eval mode.
#See help(InceptionResnetV1) for more details.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#Define a dataset and data loader
#We add the idx_to_class attribute to the dataset to enable easy recoding of label indices to identity names later one.
def collate_fn(x):
    return x[0]

#load known persons
known_people_names_path = r'embeddings\names.txt'
with open(known_people_names_path, 'rb') as file:
    known_people = pickle.load(file)
    known_people_unique = dict.fromkeys(known_people) #each name only once; create dictionary for later usage
    for k, v in known_people_unique.items():
        if v is None:
            known_people_unique[k] = 0 #init amount of names to 0

# time until values for detected people are resetted
time_of_period = 10000.0
counter_era = 0 #number of frame in the script

#need this later for reset parameters regularly (like every hour etc.)
def reset_stuff():
    threading.Timer(time_of_period, reset_stuff).start()  # called regularly
    print("Resetting values...\n")
    global counter_era #need to say global since its a global variable outside the function
    counter_era += 1
    print("We are in system era " + str(counter_era))
    for k, v in known_people_unique.items():
        known_people_unique[k] = 0  #reset to 0


#regularly reset stuff like the counter for face recognition (against overflow and also that people get re-greeted after some time like 1 hour
reset_stuff()

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
    face_analysis = False #init
    cv.imwrite(r"images_to_detect\unknown_person\frame%d.jpg" %imgcounter, frame) #save as image file
    
    # load dataset and loader form imagefolder
    dataset = datasets.ImageFolder(r'images_to_detect')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    #Perfom MTCNN facial detection
    #Iterate through the DataLoader object and detect faces and associated detection probabilities for each.
    #The MTCNN forward method returns images cropped to the detected face, if a face was detected.
    #By default only a single detected face is returned - to have MTCNN return all detected faces, set keep_all=True when creating the MTCNN object above.
    #To obtain bounding boxes rather than cropped face images, you can instead call the lower-level mtcnn.detect() function.
    list_of_aligned  = []
    unknown_person_name = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        counter = 0

        # check if aligned is empty -> no person in frame
        if x_aligned is None:
            print("There is no (known) person in frame")
            face_analysis = False #no face, then also no face analysis
            continue

        print('\nFace(s) detected with probability:')
        print(prob)
        face_analysis = True #face detected, so do face_analysis
        
        #add detected faces to list of aligned list and unknow_person_name list
        for detected_face in x_aligned:
            if detected_face is not None:
                list_of_aligned.append([detected_face])
                unknown_person_name.append([dataset.idx_to_class[y] + "_" + str(counter)])
                counter += 1

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
    counter = 0
    for face in list_of_aligned:
        face = torch.stack(face).to(device)
        unknown_embedding = resnet(face).detach().cpu()
        learned_embeddings = torch.load('embeddings\embeddings.pt')

        #Print distance matrix for classes
        dists = [(element - unknown_embedding).norm().item() for element in learned_embeddings]
        
        #create data frame
        df = pd.DataFrame(dists, columns=unknown_person_name[counter], index=known_people)
        counter += 1
        
        best_match = df.idxmin()
        print("Best match: " + best_match[0])

        #create df without the best matched person
        subdf_no_bestmatchperson = df.drop(labels=[best_match[0]], axis=0, inplace=False)

        best_match_value = df.min().values[0]
        second_best_match_value = subdf_no_bestmatchperson.min().values[0]

        #____increment entry in dictionary if face is detected AND if face is a REAL match____
        #check is best_match passed threshhold and distance to second best match is big enough
        if (best_match_value < similarity_threshold and (second_best_match_value - best_match_value) > delta_first_secon_bestmatch):
            for element in known_people_unique:
                if (element == best_match[0]):
                    known_people_unique[element] += 1
                    print("[Debugging-Info] Passed detection threshold and delta to second best match (counter was increased)")

    #reset string including the name for speech output
    current_element_for_speech_output = ''

    #identify person if recognitions succeeded several times
    #----------------------------------------------------------------
    for element in known_people_unique:
        if (known_people_unique[element] == n_counter_face_detection): #== not ">=" --> otherwise he will tell us several times; might cause bug if two instances of the same person at the same time (should not happen in reality though)
            known_people_unique[element] += 1 #increment again otherwise it might stay at n_counter_face_detection multiple times!
            speech_output_face_recognition = True
            if (current_element_for_speech_output == ''): #case distinction: because if two people triggered at the same time then only one gets greeted
                current_element_for_speech_output = element #one / first name
            else: #already at least one name in list
                current_element_for_speech_output = current_element_for_speech_output + " and " + element


    #Face Analysis including Emotion Detection
    #**********************************************
    if face_analysis == True:
        system_counter = system_counter + 1  # increment frame counter

        #dont do this every frame cause thats wasting a lot of resources
        if((system_counter % frequence_face_analysis) == 0): #only do every 8th time etc.

            # detect emotion and other parameters
            img_analysis = DeepFace.analyze(r"images_to_detect\unknown_person\frame%d.jpg" %imgcounter) #is doing analysis with one person (which one?)

            emotion = img_analysis["emotion"]
            age = img_analysis["age"]
            gender = img_analysis["gender"]
            ethnicity = img_analysis["race"]
            dominant_emotion = img_analysis["dominant_emotion"]
            emotion_rounded = {k: round(v, 2) for k, v in emotion.items()}
            ethnicity_rounded = {k: round(v, 2) for k, v in ethnicity.items()}

            #check whether emotion is repetitive: use ringbuffer: if emotion is same several times in a row then use that information
            emotion_ringbuffer.append(dominant_emotion) #add the dominant emotion to ringbuffer
            print("Emotion Ringbuffer (just for debugging)\n")
            print(emotion_ringbuffer)
            print("\n")
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


    #if requirements are fulfilled then talk
    if (speech_output_face_recognition == True):

        # speech output for person recognition
        print(CONST_BEAUTFUL_ASTERISK)
        speech_output_phrase_face_recognition = "Hey nice to see you again, " + current_element_for_speech_output + " !"
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

        if (persisting_emotion != 'neutral'): #don't say stuff if it's neutral 5 times in a row
            print(speech_output_phrase_face_analysis)
            engine.say(speech_output_phrase_face_analysis)
        speech_output_emotion_detection = False #reset
        emotion_ringbuffer.extend(['emotion1', 'emotion2', 'emotion3', 'emotion4', 'emotion5']) #reset ringbuffer

    imgcounter += 1

cap.release()
cv.destroyAllWindows()
