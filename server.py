#import face_recognition
import cv2
import numpy as np
import os
from threading import Thread
import pickle
#from websocket import create_connection
import time
import socket
import cv2
#import imagezmq
from queue import Queue
import gc
from queue import *
import collections
from datetime import timedelta
import struct
import sys
import base64
import eventlet
import socketio
from PIL import Image

import eventlet

eventlet.monkey_patch()

q_count = 0
detect_q = collections.deque(maxlen=2)
detect_time = collections.deque(maxlen=2)

people_dict = {}

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})


@sio.event
def connect(sid, environ):
    print('connect ', sid)


@sio.event
def my_message(sid, data):
    #print('message ', data)
    image_data = data['message']
    #pi_name = data['pi_name']
    frame = base64.b64decode(image_data)
    #base64_decoded = base64.b64encode(frame).decode()
    b64_src = 'data:image/jpg;base64,'
    img_src = frame
    #print("printing base64")
    with open('test.jpg', 'wb') as f_output:
        f_output.write(frame)
    sio.emit('my_response', {'pi': "pixxx"})


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


count = 0
WS = 'ws://118.185.61.235:8090'
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


def threadFrameGet(threadname, q):
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
    HOST = ''
    PORT = 5555

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()
    pi_id = [3]
    pi_id_bytes = bytes(pi_id)
    length = len(pi_id_bytes)

    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    #cap = cv2.VideoCapture('http://118.185.61.235:8000/html/cam_pic_new.php?time='+str(time_cam)+'&pDelay=40000')
    while True:
        print("MAAAAAAAAAAAAAAAAAAAAAAA")
          #  q.put(small_frame)

        #process_this_frame = not process_this_frame
        if(count % 100 == 0):
            # cap.release()
            # objgraph.show_most_common_types()
            # gc.collect()
            # objgraph.show_refs(q,filename='frame.png')
            #cap = cv2.VideoCapture('http://118.185.61.235:8000/html/cam_pic_new.php?time='+str(time_cam)+'&pDelay=40000')
            while not q.empty():
                q.get()
                # q.Queue.clear()
            continue


def recognise_person(threadname, q):
    # access names that are keys
    #global now_frame
    global q_count
    global detect_q
    global people_dict
    unlock_time = 0

    while True:
        print("inside REKO")
        frompi_frame = q.get()
        #frame = frompi_frame
        time_of_frame = time.time()
        print(time_of_frame)


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
    with open('../saved_encoding/dataset.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    queue = Queue()
    thread1 = Thread(target=threadFrameGet, args=("Thread-1", queue))
    thread2 = Thread(target=recognise_person, args=("Thread-2", queue))
    # thread3 = Thread(target=checkLock,args=("Thread-3"),queue)
    thread1.start()
    thread2.start()
    eventlet.wsgi.server(eventlet.listen(('', 5505)), app)
    #global WS
    #global ws
    #ws = create_connection(WS)
    # socket_test('Lock')
