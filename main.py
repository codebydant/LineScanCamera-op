#!/usr/bin/env python3.7
import os
import sys
import progressbar
import configparser
import numpy as np
import matplotlib.pyplot as plt

from sys import platform

## ----------------------------------------------------- System
## check if platform is windows or linux:
if platform == "linux" or platform == "linux2":
	sys.path.append("/opt/opencv-4.1.2/build/lib/python3")
elif platform == "win32":
    print("")
    sys.path.append("C:/Users/Daniel/Desktop/opencv/opencv-4.1.2/build/lib/python3")
else:
	print("only for linux or windows")
	sys.exit(-1)

##----------------------------------------------------- Opencv
import cv2

##----------------------------------------------------- Config file
## Path config.init
config_path = os.path.join(os.getcwd(), "config","config.ini")

## Read config.ini file
config = configparser.ConfigParser()
config.read(config_path)

## Class for image metadata
class ImageMetadata(object):
	'''
	A simple image object class for the region of interest and the shape of the raw image
	'''
	def __init__(self):
		self.roi = []							# Region of interest
		self.shape = {'width':0,'height':0}		# Raw image shape

		#######################################################################################
		#######################################################################################
		def __str__(self):
			return str(self.__class__) + ": " + str(self.__dict__)

## LineScanner class
class LineScanner(object):
	'''
	This class is the implementation of a LineScan Camera from an Area Scan Camera using OpenCV 4.1.2.
	In this pipeline there are two scan mode:
	1. Column mode:
	In this mode the scanner will accumulate a column of pixels from each frame in an output cv::Mat object given
	a centroid of reference. This process is much faster and smoothed but the output resolution is not too good.
	2. Width mode:
	In this mode the scanner will accumulate the region of interest in an image sequence given a centroid of reference.
	This process is more time consuming, but the output resolution is better compared to column mode.
	Note:
	The centroid is calculated from the contour of the object in the first frame.

	--Attributes--
	- Filename: 		video Filename
	- input dir:		input directory
	- output dir:		output directory
	- totalFrames:  	number of frames in the video
	- list of frames: 	List of rois for width mode scan

	--IMPORTANT--
	This class apply a rotation of 90 degrees contourclock to each frame automatically. This is due to
	the object is in horizontal position. You must be sure that the raw video is record with the object in
	horizontal position. In case you want to use a video with the object in vertical position, you will have to comment the line 208
		image_rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
	'''

	def __init__(self):
		self.filename = ""			# input filename
		self.mode = ""				# mode (column/width)
		self.input_dir = ""			# input directory
		self.output_dir = ""		# output directory
		self.totalFrames = 0		# number of frames in video
		self.list_of_Frames = []	# list of rois

	#######################################################################################
	#######################################################################################
	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

	#######################################################################################
	#######################################################################################
	def init_scan(self):
		'''
		*** DESCRIPTION
			This is the scanner function.

		*** INPUT
			<scan_mode>:		scan mode to process video e.g.(column/width)
			<video_path>:  		video file to process

		*** OUTPUT
			<scanned image>: 	A flat image (unwrapped) from the rotating object
		'''

		status = False
		if len(sys.argv) < 3:
			print("")
			print ("-> Usage %s <scan_mode> <video_file>" % sys.argv[0])
			print("")
			sys.exit(-1)

		## check if input file exists
		self.mode = sys.argv[1]
		if self.mode == "column":
			print("-> scan mode: column")
			pass
		elif self.mode == "width":
			print("-> scan mode: width")
			pass
		else:
			print("")
			print("-> mode: <column> for column pixel scan")
			print("-> mode: <width> for width_roi pixel scan")
			print ("-> Usage %s <scan_mode> <video_file>" % sys.argv[0])
			print("")
			sys.exit(-1)

		## check if input file exists
		self.input_dir = sys.argv[2]
		if not os.path.isfile(self.input_dir):
			print("-> file: %s could not be found"  % self.input_dir)
			print("")
			sys.exit(-1)

		## Save input directory
		self.filename = os.path.basename(os.path.splitext(self.input_dir)[0])
		self.input_dir = self.input_dir
		self.output_dir = os.path.join(os.getcwd(), "RESULT")
		try:
			os.mkdir(self.output_dir)
		except OSError:
			# for filename in os.listdir(self.output_dir):
			# 	file_path = os.path.join(self.output_dir, filename)
			# 	try:
			# 		if os.path.isfile(file_path) or os.path.islink(file_path):
			# 			os.unlink(file_path)
			# 		elif os.path.isdir(file_path):
			# 			shutil.rmtree(file_path)
			# 	except Exception as e:
			# 		print('Failed to delete %s. Reason: %s' % (file_path, e))
			pass

		## ================================================================ DEBUG
		## print debug info?
		if config.getboolean('DEBUG','visualize'):
			print("[DEBUG] filename:   		", self.filename)
			print("[DEBUG] input dir:  		", self.input_dir)
			print("[DEBUG] output dir: 		", self.output_dir)
		## ================================================================ DEBUG

		## Video object
		video_obj = cv2.VideoCapture(self.input_dir)

		## Total number of frames in video
		self.totalFrames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))

		## ================================================================ DEBUG
		## print debug info?
		if config.getboolean('DEBUG','visualize'):
			print("[DEBUG] total frames:        	", self.totalFrames)
			print("[DEBUG] frames per second:   	", int(video_obj.get(cv2.CAP_PROP_FPS)))
			print("[DEBUG] duration in seconds: 	", float(round(self.totalFrames/int(video_obj.get(cv2.CAP_PROP_FPS)), 2)))
		## ================================================================ DEBUG

		# totalFrames = 30
		totalFrames = self.totalFrames

		## ================================================================ DEBUG
		## print debug info?
		if config.getboolean('DEBUG','visualize'):
			print("")
			print("============================")
			print("processing video...")
			print("")
		## ================================================================ DEBUG

		if self.mode == "width":
			## scan mode based in width_roi
			self.width_scan_mode(video_obj)

			## Concadenate list of ROis
			self.concatenate_frames()

		if self.mode == "column":
			self.column_scan_mode(video_obj)


	#######################################################################################
	#######################################################################################
	def width_scan_mode(self,video_obj):

		## Create widget for progressbar
		bar = progressbar.ProgressBar(max_value=self.totalFrames,redirect_stdout=True,prefix = '-> Processing video:        ').start()

		## Plot raw image figure object
		if config.getboolean('DEFAULT','visualize'):
			fig = plt.gcf()
			fig.canvas.set_window_title('Video')

		## Centroid reference for ROi
		centroid = {'x':0,'y':0}

		## video reading frame status
		success = True
		count = 0

		## Scanner loop
		while success:

			## ================================================================ DEBUG
			## print debug info?
			if config.getboolean('DEBUG','visualize'):
				print("-----------------------------------")
				print("[DEBUG] frame {0}".format(count),end=' ')
			## ================================================================ DEBUG

			## Class object for image metadata
			image_metadata = ImageMetadata()

			## frame from video object
			success,image = video_obj.read()

			## ================================================================ DEBUG
			## Check if frame is corrupt
			if not success:
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] bad frame!")
				continue

			## Check if frame is not None
			if image is None:
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] empty frame!")
				continue

			## Check if frame is all black
			if np.sum(image) == 0:
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] black frame!")
				continue

			# Rotate 90 degrees the raw image (object from hori to vert pos)
			image_rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

			# Get image height,width
			rows,cols = image_rot.shape[:2]
			image_metadata.shape['width'] = cols
			image_metadata.shape['height'] = rows

			## Compute centroid (x,y), bounding box
			## This will use to calculate the region of interest
			if count == 0:
				boundingBoxMetadata = self.computeObjectCoordinates(image_rot)
				centroid['x'] = boundingBoxMetadata['cx']
				centroid['y'] = boundingBoxMetadata['cy']
				boundingBoxMetadata = {}

			## Get ROI based in the centroid and a predefined width roi
			## │------------------------------------│
			## │				 (cx,cy)			│<raw image height>
			## │	<cx-width roi> * <cx+width roi>	│
			## │									│
			## -------------------------------------
			ROi = image_rot[0:rows,int(centroid['x']):int(centroid['x']+3)].copy()

			image_metadata.roi = ROi

			## Save image metadata in list
			self.list_of_Frames.append(image_metadata)

			## ================================================================ PLOT
			## Visualize raw data in matplotlib
			if config.getboolean('DEFAULT','visualize'):
				## convert from BGR to RGB
				image_t = image_rot.copy()
				image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB)

				## Draw first frame
				if count > 0:

					fig.set_data(image_t)
					plt.title("frame {0}".format(count))
					plt.draw()
					plt.pause(0.001)

				## Update figure with new frame
				else:
					fig = plt.imshow(image_t)
					plt.title("frame {0}".format(count))
					plt.draw()
					plt.pause(0.001)
			## ================================================================ PLOT

			## Update frame count
			count +=1

			# if count == totalFrames:
			# 	break

			## ================================================================ DEBUG
			## print debug info?
			if config.getboolean('DEBUG','visualize'):
				print("done.")
			## ================================================================ DEBUG

			## Update progressbar
			bar.update(count)


		## print debug info?
		if config.getboolean('DEFAULT','visualize'):
			plt.close('all')

		## Close progressbar
		bar.finish()

	#######################################################################################
	#######################################################################################
	def column_scan_mode(self,video_obj):

		## Create widget for progressbar
		bar = progressbar.ProgressBar(max_value=self.totalFrames,redirect_stdout=True,prefix = '-> Processing video:        ').start()

		## Plot raw image figure object
		if config.getboolean('DEFAULT','visualize'):
			fig = plt.gcf()
			fig.canvas.set_window_title('Video')

		## Centroid reference for ROi
		centroid = {'x':0,'y':0}

		## video reading frame status
		success = True
		count = 0

		## Video dimensions
		video_height = int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
		video_width = int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))

		## Flat image column Roi version
		flatImage = np.empty((video_width,self.totalFrames,3), np.uint8)

		## combinen image for pre-visualization
		#combinedImage = np.empty((video_width,video_height+self.totalFrames,3), np.uint8)
		combinedImage = np.empty((video_width,video_height*2,3), np.uint8)

		# fig = plt.gcf()
		fig = plt.figure()
		# fig = plt.ion()
		fig.canvas.set_window_title('Video')

		## Scanner loop
		while success:

			## ================================================================ DEBUG
			## print debug info?
			if config.getboolean('DEBUG','visualize'):
				print("-----------------------------------")
				print("[DEBUG] frame {0}".format(count),end=' ')
			## ================================================================ DEBUG

			## frame from video object
			success,image = video_obj.read()

			## ================================================================ DEBUG
			## Check if frame is corrupt
			if not success:
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] bad frame!")
				continue

			## Check if frame is not None
			if image is None:
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] empty frame!")
				continue

			## Check if frame is all black
			if np.sum(image) == 0:
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] black frame!")
				continue

			# Rotate 90 degrees the raw image (object from hori to vert pos)
			image_rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

			## Compute centroid (x,y), bounding box
			## This will use to calculate the region of interest
			if count == 0:
				boundingBoxMetadata = self.computeObjectCoordinates(image_rot)
				centroid['x'] = boundingBoxMetadata['cx']
				centroid['y'] = boundingBoxMetadata['cy']
				boundingBoxMetadata = {}

			# Get image height,width
			rows,cols = image_rot.shape[:2]
			video_height = rows
			video_width = cols

			## Get column pixel based in the centroid
			column_ROi = image_rot[:,int(centroid['x'])].copy()
			# colROi = image1[:,int(image1.shape[1]/2)].copy()
			flatImage[:,count] = column_ROi

			# image.colRange(0, image.cols - 1).copyTo(combinedImage.colRange(0, image.cols - 1));
	        # lineImage.colRange(0, image.cols - 1).copyTo(combinedImage.colRange(image.cols, combinedImage.cols - 1));

			#image.colRange(0, image.cols - 1).copyTo(combinedImage.colRange(0, image.cols - 1));
			# combinedImage[:,0:video_width] = image_rot[:,0:video_width].copy()
			#combinedImage[:,self.totalFrames:combinedImage.shape[1]] = flatImage[:,0:self.totalFrames]

			# Start coordinate, here (0, 0)
			# represents the top left corner of image
			start_point = (int(centroid['x']), 0)

			# End coordinate, here (250, 250)
			# represents the bottom right corner of image
			end_point = (int(centroid['x']), video_height)

			# Green color in BGR
			color = (255, 0, 0)

			# Line thickness of 9 px
			thickness = 15

			# Using cv2.line() method
			# Draw a diagonal green line with thickness of 9 px
			image_rot_Line = cv2.line(image_rot, start_point, end_point, color, thickness)
			combinedImage[:,0:video_width] = image_rot_Line[:,0:video_width].copy()

			# newFlatVisu = np.empty((video_height,video_width,3), np.uint8)
			# newFlatVisu[:,0:self.totalFrames] = flatImage[:,0:self.totalFrames]
			# combinedImage[:,video_width:combinedImage.shape[1]] = newFlatVisu[:,0:video_width]

			combinedImage[:,video_width:video_width+self.totalFrames] = flatImage[:,0:self.totalFrames]

			cv2.namedWindow("scanner",cv2.WINDOW_NORMAL)
			cv2.resizeWindow("scanner",640,480)
			cv2.imshow("scanner",combinedImage)
			cv2.waitKey(1)

			#
			# image_t = combinedImage.copy()
			# image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB)
			#
			# image_t = cv2.resize(image_t, (480,640))
			#
			# if count > 0:
			# 	fig.set_data(image_t)
			#
			# else:
			# 	fig = plt.imshow(image_t)
			#
			# plt.title("frame {0}".format(count))
			# plt.draw()
			# plt.pause(0.0001)
			#



			## Draw first frame
			# if count > 0:
			#
			# 	fig.set_data(image_t)
			# 	plt.title("frame {0}".format(count))
			# 	plt.draw()
			# 	plt.pause(0.00001)
			#
			# ## Update figure with new frame
			# else:
			# 	fig = plt.imshow(image_t)
			# 	plt.title("frame {0}".format(count))
			# 	plt.draw()
			# 	plt.pause(0.0001)
		## ================================================================ PLOT



	        #lineImage.colRange(0, image.cols - 1).copyTo(combinedImage.colRange(image.cols, combinedImage.cols - 1));


			## ================================================================ PLOT
			## Visualize raw data in matplotlib
			if config.getboolean('DEFAULT','visualize'):
				## convert from BGR to RGB
				image_t = image_rot.copy()
				image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB)

				## Draw first frame
				if count > 0:

					fig.set_data(image_t)
					plt.title("frame {0}".format(count))
					plt.draw()
					plt.pause(0.001)

				## Update figure with new frame
				else:
					fig = plt.imshow(image_t)
					plt.title("frame {0}".format(count))
					plt.draw()
					plt.pause(0.001)
			## ================================================================ PLOT

			## Update frame count
			count +=1

			## ================================================================ DEBUG
			## print debug info?
			if config.getboolean('DEBUG','visualize'):
				print("done.")
			## ================================================================ DEBUG

			## Update progressbar
			bar.update(count)
		cv2.destroyAllWindows()
		# plt.close('all')

		## print debug info?
		if config.getboolean('DEFAULT','visualize'):
			plt.close('all')

		## Close progressbar
		bar.finish()

		dsize = (video_width,video_height)

		## resize image
		flatImage = cv2.resize(flatImage, dsize)

		## Save result in .png format
		cv2.imwrite(os.path.join(self.output_dir,self.filename+"-ColumnROi.jpg"), flatImage)
		print("")

	#######################################################################################
	#######################################################################################
	def BGRtoGray(self,image):
		if(len(image.shape)<3):
			return image
		else:
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			return image_gray

	#######################################################################################
	#######################################################################################
	def computeObjectCoordinates(self,image):
		'''
		This function is used to calculate the image coordinates of the object in an input image.

		Input: 		RGB image
		Output:		Dict {x,y,w,h,cx,cy}
		'''

		## ================================================================ DEBUG
		## print debug info?
		if config.getboolean('DEBUG','visualize'):
			print("")
			print("[DEBUG] Computing object coordinates...")
		## ================================================================ DEBUG

		## Convert ROi to grayscale
		imageGray = self.BGRtoGray(image)

		## Remove noise with gaussian filter
		imageGrayBlur = cv2.GaussianBlur(imageGray,(17,17),0)

		## threshold the grayscale image
		thresh = cv2.adaptiveThreshold(imageGrayBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

		## Remove internal/external noise in the object
		kernel = np.ones((7,7),np.uint8)
		threshOpen = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=20)
		threshClose = cv2.morphologyEx(threshOpen, cv2.MORPH_CLOSE, kernel,iterations=30)

		## ================================================================ PLOT
		## Visualize thresh data in matplotlib
		if config.getboolean('THRESH','visualize'):

			plt.suptitle("Binary threshold")

			plt.subplot(1,4,1)
			plt.title("image")
			plt.imshow(imageGray,cmap='gray')

			plt.subplot(1,4,2)
			plt.title("thresh")
			plt.imshow(thresh,cmap='gray')

			plt.subplot(1,4,3)
			plt.title("thresh open")
			plt.imshow(threshOpen,cmap='gray')

			plt.subplot(1,4,4)
			plt.title("thresh close")
			plt.imshow(threshClose,cmap='gray')

			plt.show()
		## ================================================================ PLOT

		## Erode thresh image to increase bounding area
		threshEroded = cv2.erode(threshClose, kernel, iterations=6)

		## Dilate image to reduce bounding area
		threshDilated = cv2.dilate(threshEroded, None, iterations=2)

		## Find Canny edges
		threshEdges = cv2.Canny(threshDilated, 30, 200)

		## ================================================================ PLOT
		if config.getboolean('THRESH','visualize'):

			plt.suptitle("Morphology operation")

			## plt_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
			plt.subplot(1,3,1)
			plt.title("thresh erode")
			plt.imshow(threshEroded,cmap='gray')

			plt.subplot(1,3,2)
			plt.title("thresh dilate")
			plt.imshow(threshDilated,cmap='gray')

			plt.subplot(1,3,3)
			plt.title("thresh edges")
			plt.imshow(threshEdges,cmap='gray')

			plt.show()
		## ================================================================ PLOT

		## find outer contour
		cntrs,hierarchy = cv2.findContours(threshEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		## print debug info?
		if config.getboolean('DEBUG','visualize'):
			print("[DEBUG] total contours: ",len(cntrs))

		## Sort contour list key=contourArea decreasing mode
		contours_sizes =  sorted(cntrs, key=lambda x: cv2.contourArea(x),reverse=True)

		## dict
		boundingBoxMetadata = {'x':0,'y':0,'h':0,'w':0,'cx':0,'cy':0,'angle':0}

		## Iterate list contours
		for k,contour in enumerate(contours_sizes,start=0):

			## Calculate area of the contour
			area = cv2.contourArea(contour)
			if area == 0:
				continue

			## Aprox. contour to rectangle
			arclen = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, arclen*0.005, True)

			## Image coordinates of the object
			x,y,w,h = cv2.boundingRect(contour)

			## Moment of the object
			ObjectMoment = cv2.moments(contour)
			cx = int(ObjectMoment['m10']/ObjectMoment['m00'])
			cy = int(ObjectMoment['m01']/ObjectMoment['m00'])

			## Save image coordinates on dict
			boundingBoxMetadata['x'] = x
			boundingBoxMetadata['y'] = y
			boundingBoxMetadata['w'] = w
			boundingBoxMetadata['h'] = h

			## Save centroid on dict
			boundingBoxMetadata['cx'] = cx
			boundingBoxMetadata['cy'] = cy

			## ================================================================ DEBUG
			## print debug info?
			if config.getboolean('DEBUG','visualize'):
				print("[DEBUG] bottom-left: 	",x)
				print("[DEBUG] top-left: 	",y)
				print("[DEBUG] top-right: 	",w)
				print("[DEBUG] bottom-right: 	",h)
			## ================================================================ DEBUG

			break

		## ================================================================ PLOT
		## Visualize contour data in matplotlib
		if config.getboolean('CONTOUR','visualize'):

			plt.suptitle("Detected contours")

			bgrImage = image.copy()
			rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
			rgb_t = rgbImage.copy()
			rgb_t2 = rgbImage.copy()
			cv2.drawContours(rgbImage, cntrs, -1, (0, 255, 0), 3,cv2.LINE_AA)

			plt.subplot(1,2,1)
			plt.title('ALL COUNTOURs ')
			plt.imshow(rgbImage)

			cv2.rectangle(rgb_t2,(boundingBoxMetadata['x'],boundingBoxMetadata['y']),(boundingBoxMetadata['x']+boundingBoxMetadata['w'],boundingBoxMetadata['y']+boundingBoxMetadata['h']),(0,0,255),3)

			plt.subplot(1,2,2)
			plt.title("Final contour")
			plt.imshow(rgb_t2,cmap='gray')

			plt.show()
		## ================================================================ PLOT

		return boundingBoxMetadata

	#######################################################################################
	#######################################################################################
	def concatenate_frames(self):
		'''
		In this function the stitching process is done given a number of ROis
		'''

		## Video dimensions
		video_height = self.list_of_Frames[0].shape['height']
		video_width = self.list_of_Frames[0].shape['width']

		## Flat image column Roi version
		flatImage = np.empty((video_height,len(self.list_of_Frames),3), np.uint8)

		## ================================================================ DEBUG
		## print debug info?
		if config.getboolean('DEBUG','visualize'):
			print("")
			print("============================")
			print("concadanting frames...")
			print("")
		## ================================================================ DEBUG

		## Create widget for progressbar
		bar = progressbar.ProgressBar(max_value=len(self.list_of_Frames),redirect_stdout=True,prefix = '-> concadanting frames:     ').start()

		## Flat image width Roi version
		result = []

		## Iterate over list of frames
		for n,im in enumerate(self.list_of_Frames,start=0):

			## ROi 1
			image1 = self.list_of_Frames[n].roi

			## Column ROi
			# colROi = self.list_of_Frames[n].rot[:,self.list_of_Frames[0].boundingBoxMetadata['cx']]
			# flatImage[:,n] = colROi
			colROi = image1[:,int(image1.shape[1]/2)].copy()
			flatImage[:,n] = colROi

			if(n+1 < len(self.list_of_Frames)):

				## ROi 2
				image2 = self.list_of_Frames[n+1].roi

				## ================================================================ DEBUG
				## print debug info?
				if config.getboolean('DEBUG','visualize'):
					print("[DEBUG] frame {0}-{1}".format(n,n+1))
				## ================================================================ DEBUG

				## Copy ROi1 and ROi2
				image1_roi = image1.copy()
				image2_roi = image2.copy()

				if n>0:
					## Concadenate result and new roi
					stitched_image = numpy_horizontal_concat = np.concatenate((result, image2_roi), axis=1)
				else:
					## Concadenate roi 1 and roi 2
					stitched_image = np.concatenate((image1_roi, image2_roi), axis=1)

				## Save concatenated image in result
				result = stitched_image

				## Update progressbar
				bar.update(n)

		## close progressbar
		bar.finish()

		## dsize
		dsize = (video_width,video_height)

		## resize image
		result = cv2.resize(result, dsize)

		## Save result in .png format
		cv2.imwrite(os.path.join(self.output_dir,self.filename+"-WidthROi.jpg"), result)



		## ================================================================ PLOT
		## Visualize stitched image in matplotlib
		if config.getboolean('STITCH','visualize'):

			plt.suptitle("Stitched image")

			## convert from bgr to rgb
			flatImage = cv2.cvtColor(flatImage, cv2.COLOR_BGR2RGB)
			plt.subplot(1,2,1)
			plt.title("Column ROI mode")
			plt.imshow(flatImage)

			## convert from bgr to rgb
			result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
			plt.subplot(1,2,2)
			plt.title("Width ROI mode")
			plt.imshow(result)

			plt.show()
		## ================================================================ PLOT

		print("-> output saved in: 	   ", self.output_dir)
		print("")
		return True

## Main program
if __name__ == '__main__':

	print("")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("***   MAIN PROGRAM        ***")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

	## Class object
	scanner = LineScanner()

	## Init scanner
	scanner.init_scan()
