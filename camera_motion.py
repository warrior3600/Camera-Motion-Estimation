import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Open the video file
cap = cv2.VideoCapture('/Users/ruben/Downloads/archery.mp4')

# Create an optical flow object using the Lucas-Kanade method
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

def cluster_vector(good_new):
    if len(good_new) > 0:
        labels = DBSCAN(eps=10, min_samples=5).fit(good_new)
        print(labels)
        num_objects = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        num_objects = 0
    return num_objects

# Loop through the frames in the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    
    # If there are no more frames, break out of the loop
    if not ret:
        break
    
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Estimate the optical flow vectors using the Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # Draw the motion vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    
    # Threshold the optical flow vectors
    mag, ang = cv2.cartToPolar(p1[:, 0, 0] - p0[:, 0, 0], p1[:, 0, 1] - p0[:, 0, 1])
    threshold = np.percentile(mag, 90)
    idx = np.where(mag > threshold)[0]
    good_new = good_new[idx]
    
    # Cluster the remaining optical flow vectors
    num_objects = cluster_vector(good_new)
    
    
    # Display the number of moving objects
    cv2.putText(img, f'Number of moving objects: {num_objects}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Motion Vectors', img)
    
    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Set the current frame as the previous frame for the next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()