#my environment
#python                    3.7.5
#opencv                    3.4.2 

# Standard imports
import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Read image
im = cv2.imread("Full_image.jpg", cv2.IMREAD_GRAYSCALE)

# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector()

# Detect blobs.
keypoints = detector.detect(im)

#in order to draw circles around the detected blobs
# for k in keypoints:
#     cv2.circle(im, (int(k.pt[0]), int(k.pt[1])), int(k.size), (0, 0, 255))
#
# # Show keypoints on image
# cv2.imshow("circles around blobs", im)
# cv2.waitKey(0)

#in order to crop each detected blob - and name the files by counting
i = 0
for k in keypoints:
    #k.pt is the center of the blob
    #k.size is the radius of the blob
    cv2.circle(im, (int(k.pt[0]), int(k.pt[1])), int(2), (0, 0, 255))
    ROI = im[int(k.pt[1] - k.size) : int(k.pt[1] + k.size), int(k.pt[0] - k.size) : int(k.pt[0] + k.size)]
    cv2.imwrite(str(i)+".jpg", ROI)
    i = i +1

print("done")


