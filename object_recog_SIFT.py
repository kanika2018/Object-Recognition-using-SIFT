import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth



MIN_MATCH_COUNT = 3

img1 = cv2.imread('2.jpg',0)  #Template Image
img2 = cv2.imread('1.jpg') #Search Image
img_rgb = cv2.imread('1.jpg') # trainImage
#alg = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold = 5)
alg = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = alg.detectAndCompute(img1, None)
kp2, des2 = alg.detectAndCompute(img2, None)



x = np.array([kp2[0].pt])

for i in range(len(kp2)):
    x = np.append(x, [kp2[i].pt], axis=0)

x = x[1:len(x)]

bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)


#finding clusters in search image using mean shift algorithm
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

s = [None] * n_clusters_
for i in range(n_clusters_):
    l = ms.labels_
    d, = np.where(l == i)
    #print(d.__len__())
    s[i] = list(kp2[xx] for xx in d)

des2_ = des2


#for every cluster, apply FLANN feature matching
for i in range(n_clusters_):

    kp2 = s[i]
    l = ms.labels_
    d, = np.where(l == i)
    des2 = des2_[d, ]

    if(len(kp2)<2 or len(kp1)<2):
        continue

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, 2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>3:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

        if M is None:
            print ("No Homography")
        else:
            h, w = img1.shape
        corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
        transformedCorners = cv2.perspectiveTransform(corners, M)
        
        x=int(transformedCorners[0][0][0])
        y=int(transformedCorners[0][0][1])
        #print(transformedCorners)
        #print(x)
        #print(y)
        cv2.rectangle(img_rgb, (x,y), (x+w,y+h), (0,0,255), 3)
        
        # Draw a polygon on the second image joining the transformed corners
        img2 = cv2.polylines(img2, [np.int32(transformedCorners)], True, (0, 0, 255), 2, cv2.LINE_AA)
        
    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

#display the output image with highlighted instances of template image
plt.imshow(img_rgb)
plt.show()
plt.imshow(img2)
plt.show()  
cv2.imwrite('result.jpg',img_rgb)
