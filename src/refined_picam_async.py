import cv2
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from picamera.array import PiRGBArray
from picamera import PiCamera
from sys import getsizeof
from armv7l.openvino.inference_engine import IENetwork, IEPlugin

#hacked from:
#https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
#https://opencv2-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/SingleStickSSDwithUSBCamera_OpenVINO_NCS2.py
#https://raspberrypi.stackexchange.com/questions/87062/overhead-counter

#Les Wright Dec 24 2018 (modified to support picam 30 Dec 2018)
#refined to warp speed (30 fps video, upto 28 fps inferencing 28 June 2019)

#Note cv2.dnn.blobFromImage, the size is present in the XML files, we could write a preamble to go get that data,
#Then we dont have to explicitly set it!

#Misc vars
font = cv2.FONT_HERSHEY_SIMPLEX
frameWidth = 304
frameHeight = 304
queuepulls = 0.0
detections = 0
fps = 0.0
qfps = 0.0

confThreshold = 0.4



#initialize the camera and grab a reference to the raw camera capture
#well this is interesting, we can closely match the input of the network!
#this 'seems' to have improved accuracy!
camera = PiCamera()
camera.resolution = (frameWidth,frameHeight)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(frameWidth,frameHeight)) 

# allow the camera to warmup
time.sleep(0.1)


labels_file = 'models/labels.txt'
with open(labels_file, 'r') as f:
	labels = [x.strip() for x in f]
print(labels)


#define the function that handles our processing thread
def classify_frame(inputQueue, outputQueue):
	cur_request_id = 0
	next_request_id = 1
	model_xml = "models/MobileNetSSD_deploy.xml"
	model_bin = "models/MobileNetSSD_deploy.bin"
	plugin = IEPlugin(device="MYRIAD")
	net = IENetwork(model=model_xml, weights=model_bin)
	input_blob = next(iter(net.inputs))
	out_blob = next(iter(net.outputs))
	exec_net = plugin.load(network=net, num_requests=2)
	n, c, h, w = net.inputs[input_blob].shape
	del net
	while True:
		if not inputQueue.empty():
			frame = inputQueue.get()
			in_frame = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300),127.5)
			exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
			if exec_net.requests[cur_request_id].wait(-1) == 0:
				detections = exec_net.requests[cur_request_id].outputs[out_blob]
				data_out = []
				for i in np.arange(0, detections.shape[2]):
					inference = []
					confidence = detections[0, 0, i, 2]
					if confidence > 0.2:
						idx = int(detections[0, 0, i, 1])
						box = detections[0, 0, i, 3:7] * np.array(
						[frameWidth, frameHeight, frameWidth, frameHeight])
						(startX, startY, endX, endY) = box.astype("int")
						inference.extend((idx,confidence,startX,startY,endX,endY))
						data_out.append(inference)
				outputQueue.put(data_out)
			cur_request_id, next_request_id = next_request_id, cur_request_id
			
# initialize the input queue (frames), output queue (out),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
out = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(inputQueue,outputQueue,))
p.daemon = True
p.start()

print("[INFO] starting capture...")

#time the frame rate....
timer1 = time.time()
frames = 0
queuepulls = 0
timer2 = 0
t2secs = 0

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
	if queuepulls ==1:
		timer2 = time.time()
	# Capture frame-by-frame
	frame = frame.array

	# if the input queue *is* empty, give the current frame to
	# classify
	if inputQueue.empty():
		inputQueue.put(frame)

	# if the output queue *is not* empty, grab the detections
	if not outputQueue.empty():
		out = outputQueue.get()
		queuepulls += 1



	# check to see if 'out' is not empty
	if out is not None:
		# loop over the detections
		for detection in out:
			#print(detection)
			#print("\n")
			
			objID = detection[0]
			objID = objID-1
			confidence = detection[1]

			xmin = detection[2]
			ymin = detection[3]
			xmax = detection[4]
			ymax = detection[5]

			if confidence > confThreshold:
				#bounding box
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))

				#label
				cv2.rectangle(frame, (xmin-1, ymin-1),\
				(xmin+70, ymin-10), (0,255,255), -1)
				#labeltext
				cv2.putText(frame,' '+labels[objID]+' '+str(round(confidence,2)),\
				(xmin,ymin-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)
				detections +=1 #positive detections
	


	# Display the resulting frame

	cv2.rectangle(frame, (0, 0),\
	(90, 15), (0,0,0), -1)

	cv2.putText(frame,'Threshold: '+str(round(confThreshold,1)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)


	cv2.rectangle(frame, (220, 0),\
	(300, 25), (0,0,0), -1)
	cv2.putText(frame,'VID FPS: '+str(fps), (225, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

	cv2.putText(frame,'NCS FPS: '+str(qfps), (225, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

	cv2.rectangle(frame, (0, 265),\
	(170, 300), (0,0,0), -1)
	cv2.putText(frame,'Positive detections: '+str(detections), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

	cv2.putText(frame,'Elapsed time: '+str(round(t2secs,2)), (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame',frameWidth,frameHeight)
	cv2.imshow('frame',frame)
	
	# FPS calculation
	frames += 1
	if frames >= 1:
		end1 = time.time()
		t1secs = end1-timer1
		fps = round(frames/t1secs,2)
	if queuepulls > 1:
		end2 = time.time()
		t2secs = end2-timer2
		qfps = round(queuepulls/t2secs,2)



	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	
	keyPress = cv2.waitKey(1)

	if keyPress == 113:
        	break

	if keyPress == 82:
		confThreshold += 0.1

	if keyPress == 84:
		confThreshold -= 0.1

	if confThreshold >1:
		confThreshold = 1
	if confThreshold <0:
		confThreshold = 0





cv2.destroyAllWindows()


