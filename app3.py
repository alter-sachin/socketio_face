#!/usr/bin/env python

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on available packages.
async_mode = None
import base64
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
import io
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
import face_recognition
import cv2
import numpy as np
import os
from threading import Thread
import pickle



if async_mode is None:
    try:
        import eventlet
        async_mode = 'eventlet'
    except ImportError:
        pass

    if async_mode is None:
        try:
            from gevent import monkey
            async_mode = 'gevent'
        except ImportError:
            pass

    if async_mode is None:
        async_mode = 'threading'

    print('async_mode is ' + async_mode)

# monkey patching is necessary because this application uses a background
# thread
if async_mode == 'eventlet':
    import eventlet
    eventlet.monkey_patch()
elif async_mode == 'gevent':
    from gevent import monkey
    monkey.patch_all()

import time
from threading import Thread
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None


def background_thread(threadname,q):
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        time.sleep(1)

        count += 1
        q.put(count)
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count})

def background_thread2(q):
  count = 0
  while True:
    time.sleep(1)
    count+=1
    #print(q)
    paisa = base64.b64encode(q)
    #qx =Queue()
    #qx.put(q)
    socketio.emit('my_response2',{'data':paisa})

@app.route('/')
def index():
    global thread
    global thread2
    print("loading saved encodings")
    with open('../saved_encoding/dataset.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    #queue = Queue()
    #thread1 = Thread(target=threadFrameGet, args=("Thread-1", queue))
    #thread2 = Thread(target=recognise_person, args=("Thread-2", queue))
    # thread3 = Thread(target=checkLock,args=("Thread-3"),queue)
    #if thread is None:
    #thread = Thread(target=background_thread,args=("Thread-1",queue))
    #thread2 = Thread(target=background_thread2,args=("Thread-2",queue))
    #thread.daemon = True
    ##thread2.daemon = True
    #thread.start()
    #thread2.start()
    return render_template('index.html')


@socketio.on('on_message')
def test_message(message):
    print("mil gaya")
    #session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data':"extra_data"})




@socketio.on('my_message')
def my_message(data):
    global thread

    #print('message ', data)
    image_data = data['message']  # encoded picture.
    #pi_name = data['pi_name']
    #b64_src = 'data:image/jpg;base64,'
    #b64_string = image_data.decode()
    nparr = np.fromstring(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #new_image = b64_src + image_data
    #frame = base64.b64decode(image_data)

    #base64_decframeoded = base64.b64encode(frame).decode()
    t = str(time.clock())

    #print("printing base64")
    #while counterxx < 1:
    #with open('test.jpg', 'wb') as f_output:
       #f_output.write(img)
    #print(frame)
    try:
        #frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        small_frame = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
    except Exception as e:
        print(e)

    initial_face_names = list(all_face_encodings.keys())

    total_face_encodings = np.array(list(all_face_encodings.values()))

    while True:
        print("Inside Recognition")
        print("starting to find location of person in image")
        face_locations = face_recognition.face_locations(
            rgb_small_frame, number_of_times_to_upsample=1, model="hog")
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                total_face_encodings, face_encoding, tolerance=0.44)
            name = "Unknown"


            face_distances = face_recognition.face_distance(
                total_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            second_best_match_index = np.partition(face_distances,1)[1]
            print("best_match index is"+str(face_distances[best_match_index]))
            print("second best_match value is"+str(second_best_match_index))
            diff_me = face_distances[best_match_index] - second_best_match_index
            print("difference is " + str(diff_me))
            abs_diff = abs(diff_me)





    emit('gresponse',
         {'data': 'Disconnected!', 'time': abs_diff})
    #if thread is None:
    #  thread = socketio.start_background_task(background_thread2,frame)





@socketio.on('disconnect request')
def disconnect_request():
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my response',
         {'data': 'Disconnected!', 'count': session['receive_count']})
    disconnect()


@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)


if __name__ == '__main__':
    print("loading saved encodings")
    #global all_face_encodings
    with open('saved_encoding/dataset_11july.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)
    socketio.run(app, debug=True)


