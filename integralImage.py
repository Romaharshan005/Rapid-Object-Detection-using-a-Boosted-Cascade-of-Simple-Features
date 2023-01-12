import numpy as np

def integralImage(orig_image):
	#create a new image with the same size
	integral_img = np.zeros(orig_image.shape)
	#loop through the image
	for i in range(orig_image.shape[0]):
		for j in range(orig_image.shape[1]):
			if i == 0 and j == 0:
				integral_img[i, j] = orig_image[i, j]
			elif i == 0:
				integral_img[i, j] = integral_img[i, j - 1] + orig_image[i, j]
			elif j == 0:
				integral_img[i, j] = integral_img[i - 1, j] + orig_image[i, j]
			else:
				integral_img[i, j] = integral_img[i - 1, j] + integral_img[i, j - 1] - integral_img[i - 1, j - 1] + orig_image[i, j]
	return integral_img