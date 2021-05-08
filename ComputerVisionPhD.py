import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



### Loading the data using matplotlib
img = cv2.imread("img.jpg")[:,:,::-1]

### Testing the size of the dataset
print(img.shape)

plt.imshow(img)
#plt.show()
#plt.savefig('img_orig', dpi=600)


### RGB2Gray using matplotlib
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

### directly reading the gray scale
#image = cv2.imread('img.jpg', 0)

### Testing the size of the dataset
print(img_gray.shape)

plt.imshow(img_gray, cmap = 'gray')
#plt.show()
#plt.savefig('img_gray', dpi=600)



### canny edge detection
can_edg = cv2.Canny(img_gray, 100, 200)

### Testing the size of the dataset
print(can_edg.shape)

plt.imshow(can_edg, cmap = 'gray')
#plt.show()
#plt.savefig('img_canny', dpi=600)



# modify the data type setting to 32-bit floating point
img_gray = np.float32(img_gray)



### Harris corner detection
har_cor = cv2.cornerHarris(img_gray, 2, 3, 0.04)
#har_cor = cv2.dilate(har_cor, None, iterations = 3)

### Testing the size of the dataset
print(har_cor.shape)

plt.imshow(har_cor, cmap = 'gray')
#plt.show()
#plt.savefig('img_harris', dpi=600)

