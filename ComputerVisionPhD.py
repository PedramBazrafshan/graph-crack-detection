import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sift



#################################### Loading The Data Using Matplotlib
img = cv2.imread("img.jpg")[:,:,::-1]

### Testing the size of the dataset
print('img =', img.shape)

#plt.imshow(img)
#plt.show()
#plt.savefig('img_orig', dpi=600)


#################################### RGB2Gray Using Matplotlib
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

### directly reading the gray scale
#image = cv2.imread('img.jpg', 0)

### Testing the size of the dataset
print('img_gray =', img_gray.shape)

#plt.imshow(img_gray, cmap = 'gray')
#plt.show()
#plt.savefig('img_gray', dpi=600)
#np.savetxt("img_gray.csv", img_gray, delimiter=',')



################################### Canny Edge Detection
can_edg = cv2.Canny(img_gray, 100, 200)

### Testing the size of the dataset
print('Canny =', can_edg.shape)

#plt.imshow(can_edg, cmap = 'gray')
#plt.show()
#plt.savefig('img_canny', dpi=600)
#np.savetxt("img_canny.csv", can_edg, delimiter=',')



########################## modify the data type setting to 32-bit floating point
img_gray_modif = np.float32(img_gray)



################################## Harris Corner Detection
har_cor = cv2.cornerHarris(img_gray_modif, 2, 3, 0.04)
#har_cor = cv2.dilate(har_cor, None)

### Testing the size of the dataset
print('Harris_simple =', har_cor.shape)

#plt.imshow(har_cor, cmap = 'gray')
#plt.show()
#plt.savefig('img_harris_simple', dpi=600)
#np.savetxt("img_harris_simple.csv", har_cor, delimiter=',')


################################### Corner With SubPixel Accuracy (Using Harris Corner Detection Algorithm)
har_cor_dil = cv2.dilate(har_cor, None)
ret, har_cor_thresh = cv2.threshold(har_cor_dil, 0.01*har_cor_dil.max(), 255, 0)
har_cor_thresh = np.uint8(har_cor_thresh)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(har_cor_thresh)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
har_cor_subpix = cv2.cornerSubPix(img_gray_modif, np.float32(centroids), (5,5), (-1,-1), criteria)

### Testing the size of the dataset
print('Harris_subpix =', har_cor_subpix.shape)

x_subpix = np.zeros((har_cor_subpix.shape[0], 1))
y_subpix = np.zeros((har_cor_subpix.shape[0], 1))
x_subpix[:,0] = har_cor_subpix[:, 0]
y_subpix[:,0] = har_cor_subpix[:, 1]

#plt.scatter(x_subpix, y_subpix, color= "green", marker= ".", s=30)
#plt.show()
#np.savetxt("img_harris_subpix.csv", har_cor_subpix, delimiter=',')



################################ Shi-Tomasi Corner Detection
shi_tomasi = cv2.goodFeaturesToTrack(img_gray_modif, 300, 0.01, 10)
shi_tomasi = np.int0(shi_tomasi)

### Testing the size of the dataset
print('Shi_Tomasi =', shi_tomasi.shape)

x_shi_tomasi = np.zeros((shi_tomasi.shape[0], 1))
y_shi_tomasi = np.zeros((shi_tomasi.shape[0], 1))
x_shi_tomasi[:,0] = shi_tomasi[:, 0, 0]
y_shi_tomasi[:,0] = shi_tomasi[:, 0, 1]

#plt.scatter(x_shi_tomasi, y_shi_tomasi, color= "red", marker= ".", s=30)
#plt.show()
#np.savetxt("img_shi_tomasi.csv", shi_tomasi, delimiter=',')



############################################### SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)
img_sift = img_gray
img_sift = cv2.drawKeypoints(img_gray, kp, img_sift)

### Testing the size of the dataset
print('SIFT =', des.shape)

#plt.imshow(img_sift, cmap = 'gray')
#plt.show()
#plt.savefig('img_sift', dpi=600)


