'''import face_recognition
import pickle
#import numpy as np

all_face_encodings={}
known_face_names = [
    "Divya",
    "S_Nivetha",
    "Pavi",
    "M_Nivetha"
]

#add encoding to dictionary
divya_image = face_recognition.load_image_file("./np/images/divya1.jpg")
all_face_encodings[known_face_names[0]] = face_recognition.face_encodings(divya_image)[0]

# Load a second sample picture and learn how to recognize it.
snivetha_image = face_recognition.load_image_file("./np/images/s_nivetha1.jpg")
all_face_encodings[known_face_names[1]] = face_recognition.face_encodings(snivetha_image)[0]

# Load a third sample picture and learn how to recognize it.
pavi_image = face_recognition.load_image_file("./np/images/pavi1.jpg")
all_face_encodings[known_face_names[2]] = face_recognition.face_encodings(pavi_image)[0]

# Load a second sample picture and learn how to recognize it.
mnivetha_image = face_recognition.load_image_file("./np/images/m_nivetha1.jpg")
all_face_encodings[known_face_names[3]] = face_recognition.face_encodings(mnivetha_image)[0]

#dump encoding to dat file named "dataset_faces"

with open('./np/np_dataset_faces.dat','wb') as f:
    pickle.dump(all_face_encodings,f)


import face_recognition
import pickle
import numpy as np
import cv2

#use pickle to mine through dataset

with open('./np/np_dataset_faces.dat','rb') as f:
    all_face_encodings=pickle.load(f)

#grab list of names and list of encodings

face_names = list(all_face_encodings.keys())
face_encodings = np.array(list(all_face_encodings.values()))


#unlnown face encoding 

unknown_image = face_recognition.load_image_file("./np/test_image/divya2.jpeg")
cv2.imshow('img',unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

#get result of compared faces with encoding
#results = face_recognition.compare_faces(face_encodings, unknown_encoding)

#print result with list of names with true or false
#names_with_result=list(zip(face_names,results))
#print(names_with_result)

'''



import face_recognition
import cv2
import numpy as np


student_attendance={}
divya_image = face_recognition.load_image_file("./np/images/divya1.jpg")
divya_face_encoding = face_recognition.face_encodings(divya_image)[0]

# Load a second sample picture and learn how to recognize it.
snivetha_image = face_recognition.load_image_file("./np/images/s_nivetha1.jpg")
snivetha_face_encoding = face_recognition.face_encodings(snivetha_image)[0]

# Load a third sample picture and learn how to recognize it.
pavi_image = face_recognition.load_image_file("./np/images/pavi1.jpg")
pavi_face_encoding = face_recognition.face_encodings(pavi_image)[0]

# Load a second sample picture and learn how to recognize it.
mnivetha_image = face_recognition.load_image_file("./np/images/m_nivetha1.jpg")
mnivetha_face_encoding = face_recognition.face_encodings(mnivetha_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    divya_face_encoding,
    snivetha_face_encoding,
    pavi_face_encoding,
    mnivetha_face_encoding
]
known_face_names = [
    "Divya",
    "S_Nivetha",
    "Pavi",
    "M_Nivetha"
]


#---------------------------Define Global Attendance Dictionary--------------------
for names in known_face_names:
    student_attendance[names]=[[0,0,0],"absent"]

#---------------------------attach Zoom - Teams - Meet content from server as data----------------
def capture_session_screen():
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    video_capture = cv2.VideoCapture(1)
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    frame=cv2.resize(frame,(960,540))
    #cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

    name_set=set(face_names)
    #print(name_set)
    #break

    # Release handle to the webcam
    video_capture.release()
    #cv2.destroyAllWindows()
    #cv2.waitKey(0)

    return name_set

name=capture_session_screen()
print(name)