import cv2
cap = cv2.VideoCapture('/media/ywj/Data/totalcapture/totalcapture/S1/video/freestyle3/TC_S1_freestyle3_cam1.mp4')
frames_num=cap.get(7)
print(frames_num)