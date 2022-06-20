import cv2
import numpy as np 

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = cap.read()

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(faces) == 0:
		continue

	for face in faces:
		x,y,w,h = face

		offset = 10
		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_offset,(100,100))

		cv2.imshow("Face", face_selection)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("faces",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# """
# scaleFactor – Parameter specifying how much the image size is reduced at each image scale.

# Basically the scale factor is used to create your scale pyramid. More explanation can be found here. In short, as described here, your model has a fixed size defined during training, which is visible in the xml. This means that this size of face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.

# 1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.

# minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.

# This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.
# """