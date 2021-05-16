import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



#################################### Loading The Data Using Matplotlib
img = cv.imread("img.jpg")[:,:,::-1]

### Testing the size of the dataset
print('img =', img.shape)

#plt.imshow(img)
#plt.show()
#plt.savefig('img_orig', dpi=600)


#################################### RGB2Gray Using Matplotlib
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

### directly reading the gray scale
#image = cv2.imread('img.jpg', 0)

### Testing the size of the dataset
print('img_gray =', img_gray.shape)

#plt.imshow(img_gray, cmap = 'gray')
#plt.show()
#plt.savefig('img_gray', dpi=600)
#np.savetxt("img_gray.csv", img_gray, delimiter=',')


#################################################################################################
########################################  Edge Detection ########################################
#################################################################################################


################################### Canny Edge Detection
can_edg = cv.Canny(img_gray, 100, 200)

### Testing the size of the dataset
print('Canny =', can_edg.shape)

#plt.imshow(can_edg, cmap = 'gray')
#plt.show()
#plt.savefig('img_canny', dpi=600)
#np.savetxt("img_canny.csv", can_edg, delimiter=',')



########################## modify the data type setting to 32-bit floating point
img_gray_modif = np.float32(img_gray)




################################### Image Pyramids
# generate Gaussian pyramid for the Gray-scale Image
G = img_gray.copy()
gpA = [G]
for i in range(4):
    G = cv.pyrDown(G)
    gpA.append(G)


lpA = [gpA[3]]
for i in range(3,0,-1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)


#plt.imshow(lpA[3], cmap = 'gray')
#plt.savefig('img_pyramid_simple', dpi=600)
#plt.show()



#################################################################################################
########################################  Corner Detection ######################################
#################################################################################################


################################## Harris Corner Detection
har_cor = cv.cornerHarris(img_gray_modif, 2, 3, 0.04)
#har_cor = cv2.dilate(har_cor, None)

### Testing the size of the dataset
print('Harris_simple =', har_cor.shape)

#plt.imshow(har_cor, cmap = 'gray')
#plt.show()
#plt.savefig('img_harris_simple', dpi=600)
#np.savetxt("img_harris_simple.csv", har_cor, delimiter=',')


######################### Corner With SubPixel Accuracy (Using Harris Corner Detection Algorithm)
har_cor_dil = cv.dilate(har_cor, None)
ret, har_cor_thresh = cv.threshold(har_cor_dil, 0.01*har_cor_dil.max(), 255, 0)
har_cor_thresh = np.uint8(har_cor_thresh)

# find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(har_cor_thresh)

# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
har_cor_subpix = cv.cornerSubPix(img_gray_modif, np.float32(centroids), (5,5), (-1,-1), criteria)

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
shi_tomasi = cv.goodFeaturesToTrack(img_gray_modif, 300, 0.01, 10)
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
sift = cv.xfeatures2d.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img_gray, None)
img_sift = img_gray
img_sift = cv.drawKeypoints(img_gray, kp_sift, img_sift)

### Testing the size of the dataset
print('SIFT =', des_sift.shape)

#plt.imshow(img_sift, cmap = 'gray')
#plt.show()
#plt.savefig('img_sift', dpi=600)



############################################### SURF
surf = cv.xfeatures2d.SURF_create(800)
kp_surf, des_surf = surf.detectAndCompute(img_gray, None)
img_surf = img_gray
img_surf = cv.drawKeypoints(img_gray, kp_surf, img_surf)

### Testing the size of the dataset
print('SURF =', des_surf.shape)

#plt.imshow(img_surf, cmap = 'gray')
#plt.show()
#plt.savefig('img_surf', dpi=600)




############################################### FAST
fast = cv.FastFeatureDetector_create()
kp_fast = fast.detect(img_gray, None)
img_fast = img_gray
img_fast = cv.drawKeypoints(img_gray, kp_fast, img_fast)

#plt.imshow(img_fast, cmap = 'gray')
#plt.show()
#plt.savefig('img_fast', dpi=600)



############################################### ORB
orb = cv.ORB_create(10000000)
kp_orb = orb.detect(img_gray, None)
kp_orb, des_orb = orb.compute(img_gray, kp_orb)
img_orb = img_gray
img_orb = cv.drawKeypoints(img_gray, kp_orb, img_orb)

### Testing the size of the dataset
print('ORB =', des_orb.shape)

#plt.imshow(img_orb, cmap = 'gray')
#plt.show()
#plt.savefig('img_orb', dpi=600)


