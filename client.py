import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import base64
import socketio

sio = socketio.Client()


@sio.on('connect')
def on_connect():
    print('I\'m connected!')
    sio.emit('pi_name', " ")


@sio.on('piData')
def on_message(data):
    global drone_mission_observer
    drone_mission_observer.on_next(data)


@sio.on('gresponse')
def myy_response(data):
    print(data['time'])


@sio.on('my_response2')
def myy_response(data):
    yahoo = data['data']
    framex = base64.b64decode(yahoo)
    with open('testxxx.jpg', 'wb') as f_output:
       f_output.write(framex)


@sio.on('disconnect') 
def on_disconnect(): 
    print('I\'m disconnected!') 

ws = 'http://127.0.0.1:5000'

sio.connect(ws)


cam = cv2.VideoCapture(0)
time.sleep(2);
cam.set(3, 800);
cam.set(4, 600);

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = base64.b64encode(frame)
    size = len(data)
    pi_id = [3]
    pi_id_bytes = bytes(pi_id)
    length = len(pi_id_bytes)
    total_size = size + length


    print("{}: {}".format(img_counter, size))
    sio.emit('my_message',{'message':data,'piname':"pisachin"})
    img_counter += 1

cam.release()
