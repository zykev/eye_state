from facenet_pytorch import MTCNN
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import dlib


# image_path = 'E:/onlyfat_selfa3D_2/Data/Data-S235/1100/1.jpg'
image_path = 'C:/Users/admin/Desktop/face.jpg'
predictor_model_path = 'E:/onlyfat_selfa3D_2/face_detect_align/detector/shape_predictor_68_face_landmarks.dat'
# Create face detector
detector = MTCNN(select_largest=False, post_process=False)
shape_predictor = dlib.shape_predictor(predictor_model_path)


img = cv2.imread(image_path)
#cv2.imshow('img',img)
boxes, probs = detector.detect(img)
boxes = boxes.squeeze(0)
det = dlib.rectangle(int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
face_landmarks = [(item.x, item.y) for item in shape_predictor(img, det).parts()]
face_landmarks = np.array(face_landmarks)

eye_legth = 1.6 * (face_landmarks[39,0] - face_landmarks[36,0])
eye_width = 3.5 * (face_landmarks[41,1] - face_landmarks[37,1])
eye_startx = face_landmarks[36,0] - 0.3 * eye_legth
eye_starty = face_landmarks[37,1] - 0.3 * eye_width
eye_img = img[int(eye_starty):int(eye_starty+eye_width), int(eye_startx):int(eye_startx+eye_legth)]
cv2.imshow('img', eye_img)


eye_img = cv2.resize(eye_img, (24,24))
eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', eye_img)


plt.figure(figsize=(12,8))
img = img[:, :, ::-1]
plt.imshow(img)
currentAxis=plt.gca()
rect=patches.Rectangle((boxes[0], boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect)
plt.scatter(face_landmarks[:,0], face_landmarks[:,1])
plt.show()
