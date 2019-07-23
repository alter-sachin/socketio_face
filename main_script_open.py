import face_recognition
import cv2
import numpy as np
import os
from threading import Thread
import pickle
from websocket import create_connection
import time
import socket
import cv2
import imagezmq
from queue import Queue
import gc
from queue import *
import collections
from datetime import timedelta
import struct
import sys
import base64

q_count = 0
detect_q = collections.deque(maxlen=2)
detect_time = collections.deque(maxlen=2)

people_dict = {}



count = 0
WS = 'ws://202.164.46.163:8090'
now_frame = ''
ws = ''


def on_message(message):
    print(len(str(message)))

def on_error(error):
    print(error)


def on_close():
    print("### closed ###")


def on_open():
    def run(*args):
        for i in range(3):
            time.sleep(1)
            ws.send("Hello %d" % i)
        time.sleep(1)
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


def checkLock(lst):
    if len(detect_q) == 2:
        ele = lst[0]
        chk = True
        for item in lst:
            if ele != item:
                chk = False
                break
        if (chk == True):
            print("Equal")
            return True
        else:
            print("Not Equal")
            return False
    else:
        return False
        # Comparing each element with first item  .....if one at a time....


def threadFrameGet(threadname, q,p):
    # Initialize some variables
    # create_connection(WS)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    global count
    global detect_time
   # timefrompi_frame = []
    time_cam = time.time()
    HOST=''
    PORT=5555

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn,addr=s.accept()
    pi_id = [3]

    to_client = "ktov"
    to_client_bytes = to_client.encode()
    
    pi_id_bytes = bytes(pi_id)
    length = len(pi_id_bytes)

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    #cap = cv2.VideoCapture('http://118.185.61.235:8000/html/cam_pic_new.php?time='+str(time_cam)+'&pDelay=40000')
    while True:
        person_in_question = ""
        if p.empty():
            person_in_question = ""
        else:
            person_in_question = p.get()
        person_in_question_bytes = person_in_question.encode()
        size_person = len(person_in_question_bytes)

        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(4096)
        conn.sendall(struct.pack(">L", size_person)+(person_in_question_bytes))
        #while p.empty():
        #    p.get()
        #    continue
        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        pi_data = data[:length]
        frame_data = data[length:msg_size]
        data = data[msg_size:]
        print("pi_data: {}".format(pi_data))
        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        base64_decoded = base64.b64encode(frame).decode()
        b64_src = 'data:image/jpg;base64,'
        img_src = b64_src + base64_decoded
        print("printing base64")
        #print(img_src)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        # Grab a single frame of video
        count = count + 1
        print(count)
        this_time = time.time()
        rpi_name = 0

        #frame = cv2.imread("ImageWindow.jpg")


        frame_sent_from_pi_at = rpi_name
        frame_received_at = time.time()
        time_to_receive_frame = frame_received_at - frame_sent_from_pi_at
        

        image = frame

       # timefrompi_frame.append([rpi_name, image])
        
        try:
            small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        except Exception as e:
            print(e)

        rgb_small_frame = small_frame[:, :, ::-1]
        process_this_frame = True
        
        if process_this_frame:
            print("pushing frame")
            q.put(rgb_small_frame)
         
        process_this_frame = not process_this_frame
        if(count%100 == 0):
            #cap.release()
            #objgraph.show_most_common_types()
            #gc.collect()
            #objgraph.show_refs(q,filename='frame.png')
            #cap = cv2.VideoCapture('http://118.185.61.235:8000/html/cam_pic_new.php?time='+str(time_cam)+'&pDelay=40000')
            while not q.empty():
                q.get()
                #with q.mutex:
                #    q.queue.clear()
            continue


def recognise_person(threadname, q,p):
    # access names that are keys
    #global now_frame
    global q_count
    global detect_q
    global people_dict
    unlock_time = 0
    initial_face_names = list(all_face_encodings.keys())
    for name_keys in initial_face_names:
        # initialise all values of names as 0, which means no one has been detected.
        people_dict[str(name_keys)] = 0

    print(people_dict)
    total_face_names = list(all_face_encodings.keys())
        # access values of encodings as a numpy array.
    total_face_encodings = np.array(list(all_face_encodings.values()))
    while True:
        print("inside REKO")
        frompi_frame = q.get()
        #frame = frompi_frame
        time_of_frame = time.time()
        print(time_of_frame)

        if frompi_frame is None:
            continue
        # print(frame)
        rgb_small_frame = frompi_frame
        
        # Find all the faces and face encodings in the current frame of video
        print("starting to find location of person in image")
        before_locations = time.time()
        face_locations = face_recognition.face_locations(
            rgb_small_frame, number_of_times_to_upsample=1, model="hog")
        print(face_locations)
        after_locations = time.time()
        difference_locations = after_locations - before_locations
        #print("time to find person location is"+ str(difference_locations))

        #before_encoding = time.time()
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        #after_encoding = time.time()
        #time_to_encode = after_encoding - before_encoding
        #print("time taken to encode is"+str(time_to_encode))
        # print(face_encodings)
        #global count
        #count = count + 1
        # print(count)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #print("start of comparing to encodings")
            before_compare_encoding = time.time()
            matches = face_recognition.compare_faces(
                total_face_encodings, face_encoding, tolerance=0.44)
            after_compare_encoding = time.time()
            difference_of_time_encoding = after_compare_encoding - before_compare_encoding
            #print("time to calculate comparison of encoding"+str(difference_of_time_encoding)) 

            # print(matches)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            before_face_distance = time.time() 
            face_distances = face_recognition.face_distance(
                total_face_encodings, face_encoding)
            print(face_distances)
            
            
            best_match_index = np.argmin(face_distances)
            second_best_match_index = np.partition(face_distances,1)[1]
            print("best_match index is"+str(face_distances[best_match_index]))
            print("second best_match value is"+str(second_best_match_index))
            diff_me = face_distances[best_match_index] - second_best_match_index
            print("difference is " + str(diff_me))
            abs_diff = abs(diff_me)

            after_face_distance = time.time()

            time_distance = after_face_distance - before_face_distance

            #print("time to calculate distance"+str(time_distance)) 
            time_now = time.time()
            if matches[best_match_index] and (abs_diff>.03):
                name = total_face_names[best_match_index]
                string_name = str(name)
                #people_dict[str(name)] += 1  #increasing detect by 1
                
                if(int(people_dict[string_name]) < 1):
                    #socket_test('UnLock')
                    #print(str(name) + "has less than 2 detects,will unlock at 3")
                    print(people_dict[string_name])
                    people_dict[string_name] += 1  #increasing number of detects by 1
                    print("inside if")
                    continue
                else:
                    people_dict[string_name] += 1 #increase detect by 1 , but take care
                    #socket_test('UnLock')
                    print("UnLock")
                    unlock_time = time.time()
                    time_since_pi = time_of_frame - unlock_time
                    unlock_time_str = str(unlock_time)
                    print("total time since pi sent frame to unlock"+ str(time_since_pi))
                    print("DETECTED NAME"+string_name+" SO UNLOCKING")

                    p.put(string_name)
                    
                    #name of person to display on pi screen is pushed into the p queue above that is shared between our threads.    

                    cv2.imwrite("detections2/"+string_name+unlock_time_str+".png", rgb_small_frame)
                    # reset the counter to 0.
                    people_dict[string_name] = 0
                    continue
                    # return None
        before_lock_time = time.time()
        diff_time = before_lock_time - unlock_time
        print("this is diff time"+str(diff_time))
        if(diff_time > 1.50000):
        	#socket_test('Lock')
        	print('Lock')
        	continue



def socket_test(send="Lock"):

    print("reached here")
    if ws:
        if(send == "Lock"):
            print("will lock now")
            ws.send(send)
        else:
            print("will unlock now")
            ws.send(send)


def writeFrame(frame, name):

    # write text onto the image and display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        print(name)
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 0.75, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imwrite('test1/Video' + str(count) + '.png', frame)


if __name__ == "__main__":
    # pick up the face encodings saved in the pickle file, which is saved as a dictionary.
    

    print("loading saved encodings")
    with open('saved_encoding/dataset_11july.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)


    queue = Queue()
    person_name = Queue()
    thread1 = Thread(target=threadFrameGet, args=("Thread-1", queue,person_name))
    thread2 = Thread(target=recognise_person, args=("Thread-2", queue,person_name))
    # thread3 = Thread(target=checkLock,args=("Thread-3"),queue)
    thread1.start()
    thread2.start()
    #global WS
    #global ws
    #ws = create_connection(WS)
    #socket_test('Lock')
    
    