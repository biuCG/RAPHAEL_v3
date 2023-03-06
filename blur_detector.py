# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os

def detect_blur_fft(image, size=60, thresh=10):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)
    
    # zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return (mean, mean <= thresh)

def ValidImage(path,label, thr_narrow = 0.14,thr_small = 0.003):
	"""check if object image is too small or two narrow
	returns list of imges not valid"""
	lb = np.loadtxt(os.path.join(path,label))
	img = label.split('.')[0] + '.jpg'
	im_lst = []
	# print(img)
	if (np.ndim(lb) == 1) and (lb[0] == 0):
		# print(lb)
		w = lb[3]; h = lb[4]
		if (w * h < thr_small) or () or (w / h< thr_narrow) :
			im_lst.append(img)
		elif w / h < 0.14:
			im_lst.append(img)
	else:
		cont = 0
		for coord in lb:
			# print(coord)
			if coord[0] == 0:
				w = coord[3]; h = coord[4]
				# print(w,h)
				if (cont == 0):
					img = label.split('.')[0] + '.jpg'
				else:
					img = label.split('.')[0] + str(cont + 1) + '.jpg'
				if (w * h < thr_small) or () or (w / h < thr_narrow):
						im_lst.append(img)
				cont += 1
	return im_lst