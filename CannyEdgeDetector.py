import cv2 # pip install opencv
import numpy as np
import math
from numpy import array

# Main Driverfunction
def main():
	#Enter the Absolute path of the test image.
	testImage = cv2.imread('/Users/shreekrishnatejagaraga/Downloads/zebra-crossing-1.bmp') #zebra-crossing-1
	#The below function converts the image into Grayscale where each pixel has a single value.
	grayScaleImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
	#The below function is used to obtain the dimensions of the Image pixel matrix.
	imageHeight, imageWidth = grayScaleImage.shape[:2]
	#The First Step is to remove Gaussian Noise with the help of the given 7x7 Gaussian Matrix which is performed through a call to the Gaussian Smoothing function.
	smoothImage,smoothImageMatrix,smoothImageHeight,smoothImageWidth = GaussianSmoothing(grayScaleImage,imageHeight,imageWidth)
	#The second step involves passing the output of the previous step (smoothImage) as an input to the Gradient Operator function to compute Horizontal,Vertical, Magnitude and Gradient Angle matrices.
	horizontalGradientImage,verticalGradientImage,magnitudeImage,horizontalGradientMatrix,verticalGradientMatrix,magnitudeMatrix,gradientAngleMatrix = GradientOperator(smoothImageMatrix,smoothImageHeight,smoothImageWidth)
	#The Third step depicts the call to the Non Maxima Suppression function.
	nonMaximaImage,nonMaximaImageMatrix,nonMaximaImageHeight,nonMaximaImageWidth = NonMaximaSuppression(magnitudeMatrix,gradientAngleMatrix,smoothImageHeight,smoothImageWidth)
	#The Fourth Step uses the nonMaximaImageMatrix (Magnitude matrix after non maxima suppression) as the input to perform P_Tile method Thresholding.
	thirtyPercentImage = Thresholding(nonMaximaImageMatrix,nonMaximaImageHeight,nonMaximaImageWidth,smoothImageHeight)
	#This is a visual function to display image matrices through their picture format equivalents.
	DisplayImages(smoothImage,horizontalGradientImage,verticalGradientImage,magnitudeImage,nonMaximaImage,thirtyPercentImage)

#Using the given 7x7 Gaussian Filter and also Normalising each pixel by dividing it by the total pixels(Sum of the given Gaussian Filter).
def GaussianSmoothing(grayScaleImage,imageHeight,imageWidth):
	smoothImageMatrix = []
	for i in range(0,imageHeight-6):
		smoothImageMatrix.append([])
	for i in range(0,imageHeight-6):
		for j in range(0,imageWidth-6):
			smoothImageMatrix[i].append((grayScaleImage[i][j]+grayScaleImage[i][j+1]+(2*grayScaleImage[i][j+2])+(2*grayScaleImage[i][j+3])+(2*grayScaleImage[i][j+4])+grayScaleImage[i][j+5]+grayScaleImage[i][j+6]+grayScaleImage[i+1][j]+(2*grayScaleImage[i+1][j+1])+(2*grayScaleImage[i+1][j+2])+(4*grayScaleImage[i+1][j+3])+(2*grayScaleImage[i+1][j+4])+(2*grayScaleImage[i+1][j+5])+grayScaleImage[i+1][j+6]+(2*grayScaleImage[i+2][j])+(2*grayScaleImage[i+2][j+1])+(4*grayScaleImage[i+2][j+2])+(8*grayScaleImage[i+2][j+3])+(4*grayScaleImage[i+2][j+4])+(2*grayScaleImage[i+2][j+5])+(2*grayScaleImage[i+2][j+6])+(2*grayScaleImage[i+3][j])+(4*grayScaleImage[i+3][j+1])+(8*grayScaleImage[i+3][j+2])+(16*grayScaleImage[i+3][j+3])+(8*grayScaleImage[i+3][j+4])+(4*grayScaleImage[i+3][j+5])+(2*grayScaleImage[i+3][j+6])+(2*grayScaleImage[i+4][j])+(2*grayScaleImage[i+4][j+1])+(4*grayScaleImage[i+4][j+2])+(8*grayScaleImage[i+4][j+3])+(4*grayScaleImage[i+4][j+4])+(2*grayScaleImage[i+4][j+5])+(2*grayScaleImage[i+4][j+6])+grayScaleImage[i+5][j]+(2*grayScaleImage[i+5][j+1])+(2*grayScaleImage[i+5][j+2])+(4*grayScaleImage[i+5][j+3])+(2*grayScaleImage[i+5][j+4])+(2*grayScaleImage[i+5][j+5])+grayScaleImage[i+5][j+6]+grayScaleImage[i+6][j]+grayScaleImage[i+6][j+1]+(2*grayScaleImage[i+6][j+2])+(2*grayScaleImage[i+6][j+3])+(2*grayScaleImage[i+6][j+4])+grayScaleImage[i+6][j+5]+grayScaleImage[i+6][j+6])/140)
	#The below function is used to convert the smoothImageMatrix to its Image equivalent for visual purposes while displaying the image - Unsigned Integer Format.
	smoothImage = np.array(smoothImageMatrix, dtype=np.uint8)
	smoothImageHeight,smoothImageWidth = smoothImage.shape[:2]
	#smoothImageMatrix contains the output after Gaussian Smoothing
	return smoothImage,smoothImageMatrix,smoothImageHeight,smoothImageWidth

#To Compute Gx and Gy with the corresponding Prewitt's operators to find Vertical and Horizontal Edges.
def GradientOperator(smoothImageMatrix,smoothImageHeight,smoothImageWidth):
	#Gradient along X-axis to find Vertical Edges using Prewitt's Gx Kernel
	horizontalGradientMatrix = []
	for i in range(0,smoothImageHeight-2):
		horizontalGradientMatrix.append([])
	for i in range(0,smoothImageHeight-2):
		for j in range(0,smoothImageWidth-2):
			#Normaliation done below by taking the absolute value and then dividing by 3 (0-755 max value range)
			horizontalGradientMatrix[i].append(np.absolute(-smoothImageMatrix[i][j]+smoothImageMatrix[i][j+2]-smoothImageMatrix[i+1][j]+smoothImageMatrix[i+1][j+2]-smoothImageMatrix[i+2][j]+smoothImageMatrix[i+2][j+2])/3)
	horizontalGradientImage = np.array(horizontalGradientMatrix, dtype=np.uint8)
    #Gradient along Y-axis to find Horiontal Edges using Prewitt's Gy Kernel
	verticalGradientMatrix = []
	for i in range(0,smoothImageHeight-2):
		verticalGradientMatrix.append([])
	for i in range(0,smoothImageHeight-2):
		for j in range(0,smoothImageWidth-2):
			#Normaliation done below by taking the absolute value and then dividing by 3 (0-755 max value range)
			verticalGradientMatrix[i].append(np.absolute(smoothImageMatrix[i][j]+smoothImageMatrix[i][j+1]+smoothImageMatrix[i][j+2]-smoothImageMatrix[i+2][j]-smoothImageMatrix[i+2][j+1]-smoothImageMatrix[i+2][j+2])/3)
	verticalGradientImage = np.array(verticalGradientMatrix, dtype=np.uint8)
	#Computation of Magnitude Image followed by normaliation by division by square root of 2. Math.Sqrt function is used below.
	magnitudeMatrix = []
	for i in range(0,smoothImageHeight-2):
		magnitudeMatrix.append([])
	gradientAngleMatrix = []
	for i in range(0,smoothImageHeight-2):
		gradientAngleMatrix.append([])
	for i in range(0,smoothImageHeight-2):
		for j in range(0,smoothImageWidth-2):
			magnitudeMatrix[i].append(np.sqrt((horizontalGradientMatrix[i][j]*horizontalGradientMatrix[i][j])+(verticalGradientMatrix[i][j]*verticalGradientMatrix[i][j]))/np.sqrt(2))
			#Computation of Gradient Angle Matrix for Gx = 0 Conditions
			if(horizontalGradientMatrix[i][j]== 0):
				if(verticalGradientMatrix[i][j]>0):
					gradientAngleMatrix[i].append(90)
				elif(verticalGradientMatrix[i][j]<0):
					gradientAngleMatrix[i].append(-90)
				else:
					gradientAngleMatrix[i].append(0)
			else:
				#General case computation of Grad Angle. Math.degrees converts radians to degrees while math.atan does the Tan inverse operation.
				gradientAngleMatrix[i].append(math.degrees(math.atan((verticalGradientMatrix[i][j]/horizontalGradientMatrix[i][j]))))
	magnitudeImage = np.array(magnitudeMatrix, dtype=np.uint8)
	return horizontalGradientImage,verticalGradientImage,magnitudeImage,horizontalGradientMatrix,verticalGradientMatrix,magnitudeMatrix,gradientAngleMatrix

#To perform Non-Maxima Suppression of the Magnitude matrix(magnitudeMatrix) wrt Gradient agle from gradientAngleMatrix matrix.
def NonMaximaSuppression(magnitudeMatrix,gradientAngleMatrix,smoothImageHeight,smoothImageWidth):
	k = -1
	nonMaximaImageMatrix = []
	for i in range(0,smoothImageHeight-4):
		nonMaximaImageMatrix.append([])
	for i in range(1,smoothImageHeight-3):
		k = k + 1
		for j in range(1,smoothImageWidth-3):
			#Convert negative to postive angles for easier calculation
			if(gradientAngleMatrix[i][j]<0):
				gradientAngleMatrix[i][j] = 360 + gradientAngleMatrix[i][j]
			#Sector 1
			if((gradientAngleMatrix[i][j] >=0 and gradientAngleMatrix[i][j]<=22.5 ) or (gradientAngleMatrix[i][j] > 337.5 and gradientAngleMatrix[i][j]<=360 ) or (gradientAngleMatrix[i][j] > 157.5 and gradientAngleMatrix[i][j]<= 202.5)):
				if(magnitudeMatrix[i][j] > magnitudeMatrix[i][j-1] and magnitudeMatrix[i][j] > magnitudeMatrix[i][j+1]):
					nonMaximaImageMatrix[k].append(magnitudeMatrix[i][j])
				else:
					nonMaximaImageMatrix[k].append(0)
			#Sector 2
			elif((gradientAngleMatrix[i][j] > 22.5 and gradientAngleMatrix[i][j]<=67.5 ) or (gradientAngleMatrix[i][j] > 202.5 and gradientAngleMatrix[i][j]<= 247.5)):
				if(magnitudeMatrix[i][j] > magnitudeMatrix[i-1][j+1] and magnitudeMatrix[i][j] > magnitudeMatrix[i+1][j-1]):
					nonMaximaImageMatrix[k].append(magnitudeMatrix[i][j])
				else:
					nonMaximaImageMatrix[k].append(0)
			#Sector 3
			elif((gradientAngleMatrix[i][j] > 67.5 and gradientAngleMatrix[i][j]<= 112.5 ) or (gradientAngleMatrix[i][j] > 247.5 and gradientAngleMatrix[i][j]<= 292.5)):
				if(magnitudeMatrix[i][j] > magnitudeMatrix[i-1][j] and magnitudeMatrix[i][j] > magnitudeMatrix[i+1][j]):
					nonMaximaImageMatrix[k].append(magnitudeMatrix[i][j])
				else:
					nonMaximaImageMatrix[k].append(0)
			#Sector 4
			elif((gradientAngleMatrix[i][j] > 112.5 and gradientAngleMatrix[i][j]<= 157.5 ) or (gradientAngleMatrix[i][j] > 292.5 and gradientAngleMatrix[i][j] <= 337.5)):
				if(magnitudeMatrix[i][j] > magnitudeMatrix[i-1][j] and magnitudeMatrix[i][j] > magnitudeMatrix[i+1][j]):
					nonMaximaImageMatrix[k].append(magnitudeMatrix[i][j])
				else:
					nonMaximaImageMatrix[k].append(0)
			#Output nonMaximaImageMatrix contains magnitude image after non-maxima suppression and the local maxima are set to their original values while the rest are set to 0.
	nonMaximaImage = np.array(nonMaximaImageMatrix, dtype=np.uint8)
	nonMaximaImageHeight, nonMaximaImageWidth = nonMaximaImage.shape[:2]
	return nonMaximaImage,nonMaximaImageMatrix,nonMaximaImageHeight,nonMaximaImageWidth

#P_Tile Method of Thresholding from nonMaximaImage(Magnitude image after Non-Maxima Suppression)
def thresholding(nonMaximaImageMatrix,nonMaximaImageHeight,nonMaximaImageWidth,smoothImageHeight):
	pTileList = []
	count = 0
	for i in range(0,nonMaximaImageHeight):
		for j in range(0,nonMaximaImageWidth):
			if(nonMaximaImageMatrix[i][j]!=0):
				#Creating a 1-D List of non-zero pixel values
				pTileList.append(nonMaximaImageMatrix[i][j])
	count = len(pTileList)
	#Below function is used to sort the 1-D List in descending Order.
	pTileList.sort(reverse=True)
	thirtyPercentEdgePoints = []
	for i in range(0,smoothImageHeight-4):
		thirtyPercentEdgePoints.append([])
# 30% Threshold Edge points computation.
	print("30% edge points : ")
	#Counting from Backwards for 30% Foreground pixels
	for i in range(0,math.floor(0.3*count)):
		continue
	thirtyPercentThresholdValue = round(pTileList[i-1])
	print("30% Threshold Value = ",thirtyPercentThresholdValue)
	print("Number of Edge Points with 30% Foreground = ",math.floor(0.3*count))
	for i in range(0,nonMaximaImageHeight):
		for j in range(0,nonMaximaImageWidth):
			if(nonMaximaImageMatrix[i][j]>thirtyPercentThresholdValue):
				#Highlight the Foreground pixels by setting them to 255.
				thirtyPercentEdgePoints[i].append(255)
			else:
				#Suppress the bckground Pixels
				thirtyPercentEdgePoints[i].append(0)
	thirtyPercentImage = np.array(thirtyPercentEdgePoints, dtype=np.uint8)

#Display Function to print out all the images.
def DisplayImages(smoothImage,horizontalGradientImage,verticalGradientImage,magnitudeImage,nonMaximaImage,thirtyPercentImage):
	cv2.imshow('Gaussian Image',smoothImage)
	cv2.imwrite("GaussianImage.bmp",smoothImage)
	cv2.imshow('Horizontal Gradient Image(Gx)',horizontalGradientImage)
	cv2.imwrite("Gx.bmp",horizontalGradientImage)
	cv2.imshow('Vertical Gradient Image(Gy)',verticalGradientImage)
	cv2.imwrite("Gy.bmp",verticalGradientImage)
	cv2.imshow('Magnitude Image before NMS',magnitudeImage)
	cv2.imwrite("MagnitudeImage.bmp",magnitudeImage)
	cv2.imshow('Magnitude Image after NMS',nonMaximaImage)
	cv2.imwrite("NMSImage.bmp",nonMaximaImage)
	cv2.imshow('30% Edge Points Image',thirtyPercentImage)
	cv2.imwrite("ThirtyPercentImg.bmp",thirtyPercentImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()



