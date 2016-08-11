import numpy as np
import cv2


CAMERA_NO = input('Please enter camera index or 0 if only one camera- e.g 0,1,2... :')
recordVideo = False

#get image stream from camera device 0,1,2...
cap = cv2.VideoCapture(CAMERA_NO)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
	#check if the capture is initialized
	if not cap.isOpened():
		cap.open(CAMERA_NO)

	ret, frame = cap.read()

	#if frame is read correctly
	if ret:

		#apply image smoothening
		#blur = cv2.bilateralFilter(frame,9,75,75)
		#blur = cv2.GaussianBlur(frame,(5,5),0)
		blur = cv2.medianBlur(frame,5)
		blur = cv2.blur(blur,(5,5))

		# Isolate color in frame
		#convert BGR to HSV for easier color representation
		f_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


		#Define range of colors in HSV
		lower = np.array([0,155,155])
		upper = np.array([10,255,255])

		#threshold the image
		mask = cv2.inRange(f_hsv, lower, upper)

		#reduce noise from the mask using morph transforms
		kernel = np.ones((5,5),np.uint8)
		gradient = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		#find the contours, use copy of gradient since this function manipulates
		# the original object
		image, contours, hierarchy = cv2.findContours(gradient.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		#frame = cv2.drawContours(frame, cnt, -1, (0,255,0), 3)

		#contour approximation
		for cnt in contours:
			epsilon = 0.025*cv2.arcLength(cnt,True)
			approx = cv2.approxPolyDP(cnt,epsilon,True)

			if len(approx) == 4:
				x,y,w,h = cv2.boundingRect(cnt)
				frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


		cv2.imshow('frame', frame)
		#cv2.imshow('blur',image)
		cv2.imshow('mask', gradient)

		#record video
		if recordVideo:
			out.write(frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()
