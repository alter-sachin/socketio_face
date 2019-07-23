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
    print(q)
    paisa = q
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
    image_data = data['message']
    #pi_name = data['pi_name']
    frame = base64.b64decode(image_data)
    #base64_decoded = base64.b64encode(frame).decode()
    b64_src = 'data:image/jpg;base64,'
    img_src = frame
    #print("printing base64")
    #with open('test.jpg', 'wb') as f_output:
    #   f_output.write(frame)
    if thread is None:
      thread = socketio.start_background_task(background_thread2,frame)
    #thread_sw1 = socketio.start_background_task(background_thread2,args=(frame))

    #emit('my_response', {'pi': "pixxx"})




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
    socketio.run(app, debug=True)

