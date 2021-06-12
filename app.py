from flask import Flask,render_template,request,flash,url_for,redirect,jsonify
import face_recognition
import numpy as np
import json
import time
import cv2

#import threading 

#from werkzeug.utils import secure_filename
#import json
#import random
#import tablib

###############################################
#l=learner.load_learner("./models/level1.pth") #
###############################################

app=Flask(__name__)
app.secret_key = 'h432hi5ohi3h5i5hi3o2hi'
student_attendance={}



#-----------------------Initial decleration and preperation----------------------

surendar_image = face_recognition.load_image_file("./sp/images/surendar.jpg")
surendar_face_encoding = face_recognition.face_encodings(surendar_image)[0]

# Load a second sample picture and learn how to recognize it.
vasanth_image = face_recognition.load_image_file("./sp/images/vasanth.jpg")
vasanth_face_encoding = face_recognition.face_encodings(vasanth_image)[0]

# Load a third sample picture and learn how to recognize it.
vishnu_image = face_recognition.load_image_file("./sp/images/vishnu.jpg")
vishnu_face_encoding = face_recognition.face_encodings(vishnu_image)[0]

# Load a second sample picture and learn how to recognize it.
#mnivetha_image = face_recognition.load_image_file("./np/images/m_nivetha1.jpg")
#mnivetha_face_encoding = face_recognition.face_encodings(mnivetha_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    surendar_face_encoding,
    vasanth_face_encoding,
    vishnu_face_encoding
    #mnivetha_face_encoding
]
known_face_names = [
    "Surendar",
    "Vasanth",
    "Vishnu"
    #"M_Nivetha"
]


#---------------------------Define Global Attendance Dictionary--------------------
def Initial_set():
    for names in known_face_names:
        student_attendance[names]=[[1,1,1],"present"] #change for attendance

    # Set Unknown intimation for error correction (Faces)
    student_attendance["Unknown"]=[[0,0,0],"absent"]

Initial_set()

#catch previous instance
student_attendance_pre = student_attendance

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
    #    break

    name_set=set(face_names)
    print(name_set)
    #break

    # Release handle to the webcam
    video_capture.release()
    #cv2.destroyAllWindows()
    #cv2.waitKey(0)

    return name_set

#-------------Mark Attendance--------------------------------
def mark_attendance(count):
    print("hello you called me !",count)

    # Get Name set of attended students.
    name_set=capture_session_screen()

    # Iterate over Names and mark attendance
    for name in name_set:
        student_attendance[name][0][count]=1
        if sum(student_attendance[name][0])>=2:
            student_attendance[name][1]="present"
    student_attendance_pre = student_attendance

#-------------Marked Attendance-------------------------------

#create a route
@app.route('/')
def home():
    Initial_set()
    return render_template('index.html')
@app.route('/login',methods=['GET','POST'])
def result():
    if request.method == 'POST':
        #flash(" ".join(request.form.keys()))
        #flash(" ".join(request.form.values()))
        
        duration = float(request.form["duration"])*60
        
        delay = duration/3
        print(duration,delay)
        count = 2

        while count>-1:
            #start_time = threading.Timer(120.0,mark_attendance(count))#.start()
            #start_time.start()
            time.sleep(delay)
            print("attendance marking started",count)
            mark_attendance(count)
            count-=1

        
        with open("./static/Attendance.json", "w") as outfile: 
            json.dump(student_attendance, outfile)
        #flash(student_attendance)
        #print(request.form.keys())
        #f=request.form['img_file'].split("/")
        #-----------------------------------#
        #result=jsonify(l.predict(f))        #
        #json.dump(result,"testfile.json")   #
        #-----------------------------------#
        #with open("testfile.json") as jfile:
         #   dicl=json.load(jfile)
        #ifile=f[len(f)-1]
        #if ifile in dicl.keys():
        #    result=dicl[ifile]
        #furl="/test_images/"+f[len(f)-1]

        #attendance=student_attendance.values(),
        return render_template('login.html')
        #return render_template('index.html',isindex=True,len=len(student_attendance),attendance=list(student_attendance.values()),names=list(student_attendance.keys()))
        #return jsonify(student_attendance)
    else:
        return render_template('login.html')
        #return render_template('index.html',isindex=True,len=len(student_attendance_pre),attendance=list(student_attendance_pre.values()),names=list(student_attendance_pre.keys()))
        #return redirect(url_for('home'))

@app.route('/attendance',methods=['GET','POST'])
def attendance():
    if request.method == 'POST':
        #flash(" ".join(request.form.keys()))
        #flash(" ".join(request.form.values()))
        if request.form["username"] == "admin" and request.form["password"]== "admin123":
            return render_template('index.html',isindex=True,len=len(student_attendance),attendance=list(student_attendance.values()),names=list(student_attendance.keys()))
        else:
            flash("Invalid User Name and Password.")
            return redirect(url_for('result'))
    else:
        return redirect(url_for('home'))

@app.route('/api')
def model():
    return jsonify(student_attendance)
    
