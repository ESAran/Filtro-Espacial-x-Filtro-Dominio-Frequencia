import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import logging as log

# img = cv2.imread('t1.jpg', 0)

# cv2.imshow('img',img)

img = cv2.imread(r'img/t1.jpg', 0)
img = cv2.imread(r'img/ideallowpass.png', 0)
img = cv2.imread(r'img/messi.jpg', 0)

def createPB(shape, center, radius, lpType=2, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # ideal low-pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 1
        lpFilter[lpFilter >= pow(radius, 2.0)] = 0
    elif lpType == 1: #Butterworth low-pass filter 
        lpFilter = 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == 2: # Gaussian low pass filter
        lpFilter = np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix

def createPA(shape, center, radius, lpType=2, n=2):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)
    lpFilter_matrix = np.zeros(shape, np.float32)
    if lpType == 0:  # Ideal high pass filter
        lpFilter = np.copy(d)
        lpFilter[lpFilter < pow(radius, 2.0)] = 0
        lpFilter[lpFilter >= pow(radius, 2.0)] = 1
    elif lpType == 1: #Butterworth Highpass Filters 
        lpFilter = 1.0 - 1.0 / (1 + np.power(np.sqrt(d)/radius, 2*n))
    elif lpType == 2: # Gaussian Highpass Filter 
        lpFilter = 1.0 - np.exp(-d/(2*pow(radius, 2.0)))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix

def add_padding(img, padding_height, padding_width):
    n, m = img.shape

    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height : n + padding_height, padding_width : m + padding_width] = img

    return padded_img

def conv2d_sharpening(img, kernel, padding=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape  # Atribui valor à variável k_height, k_width

    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = k_height // 2  # Atribui valor à variável pad_height
    pad_width = k_width // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Atribui valor à variável padded_img

    #print(padded_img)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  # Atribui valor à variável output

    # Perform convolution
    for i_img in range(img_height):  # Loop usando i
        for j_img in range(img_width):  # Loop usando j
            #calcula kernel
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    output[i_img, j_img] = output[i_img, j_img] + (padded_img[i_img+i_kernel, j_img+j_kernel] * kernel[i_kernel, j_kernel])  # Atribui valor à variável output[i, j]
            output[i_img, j_img] = int(output[i_img, j_img])

    return np.array(output, dtype=np.float32)

def gauss_create(sigma=1, size_x=3, size_y=3):
    '''
    Create normal (gaussian) distribuiton
    '''
    x, y = np.meshgrid(np.linspace(-1,1,size_x), np.linspace(-1,1,size_y))
    calc = 1/((2*np.pi*(sigma**2)))
    exp = np.exp(-(((x**2) + (y**2))/(2*(sigma**2))))

    return exp*calc

def conv2d(img, kernel, padding=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape  # Atribui valor à variável k_height, k_width

    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = k_height // 2  # Atribui valor à variável pad_height
    pad_width = k_width // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Atribui valor à variável padded_img

    print(padded_img.shape)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  # Atribui valor à variável output

    # Perform convolution
    for i_img in range(img_height):  # Loop usando i
        for j_img in range(img_width):  # Loop usando j
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    output[i_img, j_img] = output[i_img, j_img] + (padded_img[i_img+i_kernel, j_img+j_kernel] * kernel[i_kernel, j_kernel])  # Atribui valor à variável output[i, j]
            output[i_img, j_img] = int(output[i_img, j_img])

    return np.array(output, dtype=np.uint8)

def MSE(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def RMSE(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return np.sqrt(err)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

image_f32 = np.float32(img) # convert from uint8 into float32
dft = cv2.dft(image_f32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

print("Imagem original")
cv2.imshow('Imagem original', img)
time.sleep(5)
cv2.destroyAllWindows()

#Esmaecimento Gaussiano
print('esmaecimento gaussiano')
gaus_3x3 = gauss_create(sigma=1, size_x=3, size_y=3)
mean_3x3 = np.ones((3, 3))/9
GaussianImage = conv2d(image_f32, gaus_3x3)
mse_gauss = MSE(GaussianImage, image_f32)
rmse_gauss = RMSE(GaussianImage, image_f32)
psnr_gauss =  PSNR(GaussianImage, image_f32)

nrows, ncols = dft_shift.shape[:2]
real = np.power(dft_shift[:, :, 0], 2.0)
imaginary = np.power(dft_shift[:, :, 1], 2.0)
amplitude = np.sqrt(real+imaginary) 
minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(amplitude)

#Passa-baixa Ideal
print('passa-baixa ideal')
maskPB = createPB(dft_shift.shape, center=(int(ncols/2), int(nrows/2)), radius=35, lpType=0, n=2) #Quanto maior o radius aqui, menos borrada a imagem fica, mas com métricas melhores ex: 80, psnr > 30 e rmse < 7, porém menos blur
#maskPB = createPB(dft_shift.shape, center=maxLoc, radius=35, lpType=1, n=2)
filtered_freqPB = dft_shift*maskPB
f_ishiftPB = np.fft.ifftshift(filtered_freqPB)  #inversa da fft
img_backPB = cv2.idft(f_ishiftPB)        #inversa da dft
img_backPB = cv2.magnitude(img_backPB[:,:,0],img_backPB[:,:,1])  #recuperando a imagem capturando a magnitude (intesidade) 
img_backPB = np.array(img_backPB, dtype=np.float32)

#Normalizando a imagem
print('normalizando a imagem')
img_backPB -= img_backPB.min()
img_backPB = img_backPB * 255 / img_backPB.max()
img_backPB = img_backPB.astype(np.uint8)

#Valores das métricas
print('valores das métricas')
mse_PB = MSE(img_backPB, image_f32)
rmse_PB = RMSE(img_backPB, image_f32)
psnr_PB =  PSNR(img_backPB, image_f32)

#Passa-baixa Gaussiano
print('passa-baixa gaussiano')
d0PBG = 35 # quanto maior menor o blur
maskPBG = createPB(dft_shift.shape, center=(int(ncols/2), int(nrows/2)), radius=d0PBG, lpType=2, n=2) 
filtered_freqPBG = dft_shift*maskPBG
f_ishiftPBG = np.fft.ifftshift(filtered_freqPBG)  #inversa da fft
img_backPBG = cv2.idft(f_ishiftPBG)        #inversa da dft
img_backPBG = cv2.magnitude(img_backPBG[:,:,0],img_backPBG[:,:,1])  #recuperando a imagem capturando a magnitude (intesidade)

#Normalizando a img
print('normalizando a imagem')
img_backPBG = np.array(img_backPBG, dtype=np.float32)
img_backPBG -= img_backPBG.min()
img_backPBG = img_backPBG * 255 / img_backPBG.max()
img_backPBG = img_backPBG.astype(np.uint8)

mse_PBG = MSE(img_backPBG, image_f32)
rmse_PBG = RMSE(img_backPBG, image_f32)
psnr_PBG =  PSNR(img_backPBG, image_f32)

#plotando sem regularização com matplotlib Gauss x Passa-Baixa Ideal x Passa-Baixa Gaussiano
print('plotando sem regularização com matplotlib Gauss x Passa-Baixa Ideal x Passa-Baixa Gaussiano')
plt.subplot(131),plt.imshow(GaussianImage, cmap = 'gray')
time.sleep(5)
plt.title(f'Gaussiano\nMSE: {mse_gauss:.2f}\nPSNR: {psnr_gauss:.2f}\nRMSE: {rmse_gauss:.2f}'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_backPB, cmap = 'gray')
time.sleep(5)
plt.title(f'Passa-Baixa Ideal\nMSE: {mse_PB:.2f}\nPSNR: {psnr_PB:.2f}\nRMSE: {rmse_PB:.2f}'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_backPBG, cmap = 'gray')
time.sleep(5)
plt.title(f'Passa-Baixa Gaussiano (d0={d0PBG})\nMSE: {mse_PBG:.2f}\nPSNR: {psnr_PBG:.2f}\nRMSE: {rmse_PBG:.2f}'), plt.xticks([]), plt.yticks([])
plt.show()
time.sleep(25)

#Canny
print("canny")
canny_img = cv2.Canny(img,100,200)
cv2.imshow('Canny na imagem original', canny_img)
time.sleep(25)
cv2.destroyAllWindows()

#Normalizando a img
print("normalizando a imgem")
canny_img = np.array(canny_img, dtype=np.float32)
canny_img -= canny_img.min()
canny_img = canny_img * 255 / canny_img.max()
canny_img = canny_img.astype(np.uint8)

#Sobel
print("sobel")
kernel_sobel_1 = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
kernel_sobel_2 = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
img_sobel_1 = conv2d_sharpening(img, kernel_sobel_1)
img_sobel_2 = conv2d_sharpening(img, kernel_sobel_2)
img_sobel = np.abs(img_sobel_1)+np.abs(img_sobel_2)

#Normalizando a img
print("normalizando a imagem")
img_sobel = np.array(img_sobel, dtype=np.float32)
img_sobel -= img_sobel.min()
img_sobel = img_sobel * 255 / img_sobel.max()
img_sobel = img_sobel.astype(np.uint8)

#Valores métricas
print("valores metricas")
mse_sobel = MSE(img_sobel, canny_img)
rmse_sobel = RMSE(img_sobel, canny_img)
psnr_sobel =  PSNR(img_sobel, canny_img)

#Passa-Alta Ideal
print("passa alta ideal")
maskPA = createPA(dft_shift.shape, center=(int(ncols/2), int(nrows/2)), radius=35, lpType=0, n=2)
filtered_freq = dft_shift*maskPA
f_ishift = np.fft.ifftshift(filtered_freq)  #inversa da fft
img_backPA = cv2.idft(f_ishift)        #inversa da dft
img_backPA = cv2.magnitude(img_backPA[:,:,0],img_backPA[:,:,1])  #recuperando a imagem capturando a magnitude (intesidade) 
img_backPA = np.array(img_backPA, dtype=np.float32)

#Normalizando a img
print("normalizando a imagem")
img_backPA = np.array(img_backPA, dtype=np.float32)
img_backPA -= img_backPA.min()
img_backPA = img_backPA * 255 / img_backPA.max()
img_backPA = img_backPA.astype(np.uint8)

#Valores métricas
print("valores metricas")
mse_backPA = MSE(img_backPA, canny_img)
rmse_backPA = RMSE(img_backPA, canny_img)
psnr_backPA =  PSNR(img_backPA, canny_img)

#Passa-Alta Gaussiano
print("passa alto gaussiano")
d0PAG = 80
maskPAG = createPA(dft_shift.shape, center=(int(ncols/2), int(nrows/2)), radius=d0PAG, lpType=2, n=2) #d0 = 80
filtered_freq = dft_shift*maskPAG
f_ishiftPAG = np.fft.ifftshift(filtered_freq)  #inversa da fft
img_backPAG = cv2.idft(f_ishiftPAG)        #inversa da dft
img_backPAG = cv2.magnitude(img_backPAG[:,:,0],img_backPAG[:,:,1])  #recuperando a imagem capturando a magnitude (intesidade) 
img_backPAG = np.array(img_backPAG, dtype=np.float32)

#Normalizando a img
print("normalizando a imagem")
img_backPAG = np.array(img_backPAG, dtype=np.float32)
img_backPAG -= img_backPAG.min()
img_backPAG = img_backPAG * 255 / img_backPAG.max()
img_backPAG = img_backPAG.astype(np.uint8)

#Valores métricas
mse_backPAG = MSE(img_backPAG, canny_img)
rmse_backPAG = RMSE(img_backPAG, canny_img)
psnr_backPAG =  PSNR(img_backPAG, canny_img)


#plotando sem regularização com matplotlib
print("plotando sem regularização com matplotlib")
plt.subplot(131),plt.imshow(img_backPA, cmap = 'gray')
time.sleep(5)
plt.title(f'Passa-Alta Ideal\nMSE: {mse_backPA:.2f}\nPSNR: {psnr_backPA:.2f}\nRMSE: {rmse_backPA:.2f}'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_backPAG, cmap = 'gray')
time.sleep(5)
plt.title(f'Passa-Alta Gaussiano d0 = {d0PAG}\n\nMSE: {mse_backPAG:.2f}\nPSNR: {psnr_backPAG:.2f}\nRMSE: {rmse_backPAG:.2f}'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_sobel, cmap = 'gray')
time.sleep(5)
plt.title(f'Sobel\nMSE: {mse_sobel:.2f}\nPSNR: {psnr_sobel:.2f}\nRMSE: {rmse_sobel:.2f}'), plt.xticks([]), plt.yticks([])
plt.show()
time.sleep(25)
input("")
