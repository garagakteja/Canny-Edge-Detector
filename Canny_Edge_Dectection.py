import cv2
import numpy as np
import math
from numpy  import array

# Main function Definition

def main():
	#Enter the full path of the test image in the below line. 
	img = cv2.imread('/Users/shreekrishnatejagaraga/Downloads/zebra-crossing-1.bmp') #zebra-crossing-1
	#The below function converts the image into Grayscale where each pixel has a single value.
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#The below function is used to obtain the dimensions of the Image pixel matrix.
	height, width = gray_img.shape[:2]
	#The First Step to remove Gaussian Noise with the help of the given 7x7 Gaussian Matrix, is performed through a call to Gaussian_Smoothing
	a,hor_img,new_height,new_width = Gaussian_smoothing(gray_img,height,width)
	#The second step involves passing the output of the previous step (hor_img) as an input to Gradient_operator to compute Gx,Gy,Gmag. and Ggradient_angle matrices.
	b,c,d,x_image,y_image,mag_image,grad_angle = Gradient_operator(hor_img,new_height,new_width)
	#The third step below depicts the call to non_maxima_suppression with mag_image(Gmag.) and grad_angle(Ggradient_angle) as input matrices.
	e,non_mag_image,non_height,non_width = non_maxima_suppression(mag_image,grad_angle,new_height,new_width)
	#The Fourth Step uses the non_mag_image (Magnitude matrix after non maxima suppression) as the input to perform P_Tile method Thresholding.
	f,g,h = thresholding(non_mag_image,non_height,non_width,new_height)
	#This is a visual function to display image matrices through their picture format equivalents.
	Output_Display(hor_img,a,b,c,d,e,f,g,h)


#Using the given 7x7 Gaussian Filter and also normaliing each pixel by diving the total pixel by 140(Sum of the given Gaussian Filter)
def Gaussian_smoothing(gray_img,height,width):
	hor_img = []
	for i in range(0,height-6):
		hor_img.append([])
	for i in range(0,height-6):
		for j in range(0,width-6):
			hor_img[i].append((gray_img[i][j]+gray_img[i][j+1]+(2*gray_img[i][j+2])+(2*gray_img[i][j+3])+(2*gray_img[i][j+4])+gray_img[i][j+5]+gray_img[i][j+6]+gray_img[i+1][j]+(2*gray_img[i+1][j+1])+(2*gray_img[i+1][j+2])+(4*gray_img[i+1][j+3])+(2*gray_img[i+1][j+4])+(2*gray_img[i+1][j+5])+gray_img[i+1][j+6]+(2*gray_img[i+2][j])+(2*gray_img[i+2][j+1])+(4*gray_img[i+2][j+2])+(8*gray_img[i+2][j+3])+(4*gray_img[i+2][j+4])+(2*gray_img[i+2][j+5])+(2*gray_img[i+2][j+6])+(2*gray_img[i+3][j])+(4*gray_img[i+3][j+1])+(8*gray_img[i+3][j+2])+(16*gray_img[i+3][j+3])+(8*gray_img[i+3][j+4])+(4*gray_img[i+3][j+5])+(2*gray_img[i+3][j+6])+(2*gray_img[i+4][j])+(2*gray_img[i+4][j+1])+(4*gray_img[i+4][j+2])+(8*gray_img[i+4][j+3])+(4*gray_img[i+4][j+4])+(2*gray_img[i+4][j+5])+(2*gray_img[i+4][j+6])+gray_img[i+5][j]+(2*gray_img[i+5][j+1])+(2*gray_img[i+5][j+2])+(4*gray_img[i+5][j+3])+(2*gray_img[i+5][j+4])+(2*gray_img[i+5][j+5])+gray_img[i+5][j+6]+gray_img[i+6][j]+gray_img[i+6][j+1]+(2*gray_img[i+6][j+2])+(2*gray_img[i+6][j+3])+(2*gray_img[i+6][j+4])+gray_img[i+6][j+5]+gray_img[i+6][j+6])/140)
	#The below function is used to convert the hor_img matrix to its Image equivalent for visual purposes while displaying the image - Unsigned Integer Format.
	a = np.array(hor_img, dtype=np.uint8)
	new_height, new_width = a.shape[:2]
	#hor_img contains the output after Gaussian Smoothing
	return a,hor_img,new_height,new_width

#To Compute Gx and Gy with the corresponding Prewitt's operators to find Vertical and Horizontal Edges.
def Gradient_operator(hor_img,new_height,new_width):
	#Gradient along X-axis to find Vertical Edges using Prewitt's Gx Kernel
	x_image = []
	for i in range(0,new_height-2):
		x_image.append([])
	for i in range(0,new_height-2):
		for j in range(0,new_width-2):
			#Normaliation done below by taking the absolute value and then dividing by 3 (0-755 max value range)
			x_image[i].append(np.absolute(-hor_img[i][j]+hor_img[i][j+2]-hor_img[i+1][j]+hor_img[i+1][j+2]-hor_img[i+2][j]+hor_img[i+2][j+2])/3)
	b = np.array(x_image, dtype=np.uint8)

	#Gradient along Y-axis to find Horiontal Edges using Prewitt's Gy Kernel
	y_image = []
	for i in range(0,new_height-2):
		y_image.append([])

	for i in range(0,new_height-2):
		for j in range(0,new_width-2):
			#Normaliation done below by taking the absolute value and then dividing by 3 (0-755 max value range)
			y_image[i].append(np.absolute(hor_img[i][j]+hor_img[i][j+1]+hor_img[i][j+2]-hor_img[i+2][j]-hor_img[i+2][j+1]-hor_img[i+2][j+2])/3)
	c = np.array(y_image, dtype=np.uint8)
	#Computation of Magnitude Image followed by normaliation by division by square root of 2. Math.Sqrt function is used below.
	mag_image = []
	for i in range(0,new_height-2):
		mag_image.append([])

	grad_angle = []
	for i in range(0,new_height-2):
		grad_angle.append([])
	for i in range(0,new_height-2):
		for j in range(0,new_width-2):
			mag_image[i].append(np.sqrt((x_image[i][j]*x_image[i][j])+(y_image[i][j]*y_image[i][j]))/np.sqrt(2))
			#Computation of Gradient Angle Matrix for Gx = 0 Conditions
			if(x_image[i][j]== 0):
				if(y_image[i][j]>0):
					grad_angle[i].append(90)
				elif(y_image[i][j]<0):
					grad_angle[i].append(-90)
				else:
					grad_angle[i].append(0)
			else:
				#General case computation of Grad Angle. Math.degrees converts radians to degrees while math.atan does the Tan inverse operation.
				grad_angle[i].append(math.degrees(math.atan((y_image[i][j]/x_image[i][j]))))
	d = np.array(mag_image, dtype=np.uint8)
	return b,c,d,x_image,y_image,mag_image,grad_angle

#To perform Non-Maxima Suppression of the Magnitude matrix(mag_image) wrt Gradient agle from grad_angle matrix.
def non_maxima_suppression(mag_image,grad_angle,new_height,new_width):
	new_i = -1
	non_mag_image = []
	for i in range(0,new_height-4):
		non_mag_image.append([])
	for i in range(1,new_height-3):
		new_i = new_i + 1
		for j in range(1,new_width-3):
			#Convert negative to postive angles for easier calculation
			if(grad_angle[i][j]<0):
				grad_angle[i][j] = 360 + grad_angle[i][j]
			#Sector 1
			if((grad_angle[i][j] >=0 and grad_angle[i][j]<=22.5 ) or (grad_angle[i][j] > 337.5 and grad_angle[i][j]<=360 ) or (grad_angle[i][j] > 157.5 and grad_angle[i][j]<= 202.5)):
				if(mag_image[i][j] > mag_image[i][j-1] and mag_image[i][j] > mag_image[i][j+1]):
					non_mag_image[new_i].append(mag_image[i][j])
				else:
					non_mag_image[new_i].append(0)
			#Sector 2
			elif((grad_angle[i][j] > 22.5 and grad_angle[i][j]<=67.5 ) or (grad_angle[i][j] > 202.5 and grad_angle[i][j]<= 247.5)):
				if(mag_image[i][j] > mag_image[i-1][j+1] and mag_image[i][j] > mag_image[i+1][j-1]):
					non_mag_image[new_i].append(mag_image[i][j])
				else:
					non_mag_image[new_i].append(0)
			#Sector 3
			elif((grad_angle[i][j] > 67.5 and grad_angle[i][j]<= 112.5 ) or (grad_angle[i][j] > 247.5 and grad_angle[i][j]<= 292.5)):
				if(mag_image[i][j] > mag_image[i-1][j] and mag_image[i][j] > mag_image[i+1][j]):
					non_mag_image[new_i].append(mag_image[i][j])
				else:
					non_mag_image[new_i].append(0)
			#Sector 4
			elif((grad_angle[i][j] > 112.5 and grad_angle[i][j]<= 157.5 ) or (grad_angle[i][j] > 292.5 and grad_angle[i][j] <= 337.5)):
				if(mag_image[i][j] > mag_image[i-1][j] and mag_image[i][j] > mag_image[i+1][j]):
					non_mag_image[new_i].append(mag_image[i][j])
				else:
					non_mag_image[new_i].append(0)
			#Output non_mag_image contains magnitude image after non-maxima suppression and the local maxima are set to their original values while the rest are set to 0.
	e = np.array(non_mag_image, dtype=np.uint8)
	non_height, non_width = e.shape[:2]
	return e,non_mag_image,non_height,non_width

#P_Tile Method of Thresholding from non_mage_image(magnitude image after non-maxima suppression)
def thresholding(non_mag_image,non_height,non_width,new_height):
	p_tile_list = []
	count = 0
	for i in range(0,non_height):
		for j in range(0,non_width):
			if(non_mag_image[i][j]!=0):
				#Creating a 1-D List of non-zero pixel values
				p_tile_list.append(non_mag_image[i][j])
	count = len(p_tile_list)
	#Below function is used to sort the 1-D List in descending Order.
	p_tile_list.sort(reverse=True)
	final_ten = []
	for i in range(0,new_height-4):
		final_ten.append([])
#----------------------------------------------------------------------- 10% Threshold
	print("10 edge points : ")
	#Counting from Backwards for 10% Foreground pixels
	for i in range(0,math.floor(0.1*count)):
		continue
	#Threshold Value at the boundary of 10% i-1 since array values start from 0th index.
	threshold_10 = round(p_tile_list[i-1])
	print("10% Threshold Value = ",threshold_10)
	print("Number of Edge Points with 10 Foreground = ",math.floor(0.1*count))
	for i in range(0,non_height):
		for j in range(0,non_width):
			if(non_mag_image[i][j]>threshold_10):
				#Highlight the Foreground pixels by setting them to 255.
				final_ten[i].append(255)
			else:
				#Suppress the bckground Pixels
				final_ten[i].append(0)
	f = np.array(final_ten, dtype=np.uint8)

#------------------------------------------------------------------------ 30% Threshold
	final_thirty = []
	for i in range(0,new_height-4):
		final_thirty.append([])
	print("30 edge points : ")
	#Counting from Backwards for 30% Foreground pixels
	for i in range(0,math.floor(0.3*count)):
		continue
	threshold_30 = round(p_tile_list[i-1])
	print("30% Threshold Value = ",threshold_30)
	print("Number of Edge Points with 30 Foreground = ",math.floor(0.3*count))
	for i in range(0,non_height):
		for j in range(0,non_width):
			if(non_mag_image[i][j]>threshold_30):
				#Highlight the Foreground pixels by setting them to 255.
				final_thirty[i].append(255)
			else:
				#Suppress the bckground Pixels
				final_thirty[i].append(0)
	g = np.array(final_thirty, dtype=np.uint8)



#------------------------------------------------------------------------- 50% Threshold

	final_fifty = []
	for i in range(0,new_height-4):
		final_fifty.append([])
	print("50 edge points : ")
	#Counting from Backwards for 50% Foreground pixels
	for i in range(0,math.floor(0.5*count)):
		continue
	threshold_50 = round(p_tile_list[i-1])
	print("50% Threshold Value = ",threshold_50)
	print("Number of Edge Points with 50 Foreground = ",math.floor(0.5*count))
	for i in range(0,non_height):
		for j in range(0,non_width):
			if(non_mag_image[i][j]>threshold_50):
				#Highlight the Foreground pixels by setting them to 255.
				final_fifty[i].append(255)
			else:
				#Suppress the bckground Pixels
				final_fifty[i].append(0)
	h = np.array(final_fifty, dtype=np.uint8)
	return f,g,h



#------------------------------------------------------------------------- Below Display Function

def Output_Display(hor_img,a,b,c,d,e,f,g,h):
	cv2.imshow('Gaussian Image',a)
	cv2.imwrite("Gaussian_Image.bmp",a)
	cv2.imshow('Gx',b)
	cv2.imwrite("Gx.bmp",b)
	cv2.imshow('Gy',c)
	cv2.imwrite("Gy.bmp",c)
	cv2.imshow('Magnitude Image before NMS',d)
	cv2.imwrite("Magnitude_Image.bmp",d)
	cv2.imshow('Magnitude Image after NMS',e)
	cv2.imwrite("NMS_Magnitude_Image.bmp",e)
	cv2.imshow('50%',h)
	cv2.imwrite("Fifty_Percent.bmp",h)
	cv2.imshow('30%',g)
	cv2.imwrite("Thirty_Percent.bmp",g)
	cv2.imshow('10%',f)
	cv2.imwrite("Ten_Percent.bmp",f)
	#cv2.imshow('mag_image',d)
	#cv2.imshow('my_image1',gray)"""
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()



