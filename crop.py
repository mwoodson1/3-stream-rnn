import cv2
import numpy as np
import scipy.io as io

output = []
VIDEO_FILE = 'UCF-101/BasketballDunk/v_BasketballDunk_g01_c01.avi'
#VIDEO_FILE = 'drop.avi'
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(VIDEO_FILE)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(VIDEO_FILE,fourcc, 20.0, (640,480))

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret==True:
        # Process the frame
        print frame.shape
        frame_height, frame_width, RGB = frame.shape
        
        height_offset = frame_height/4
        width_offset = frame_width/4

        new_frame = np.zeros((frame_height/2,frame_width/2, 3))
        print new_frame.shape
        
        for i in range(frame_height/2):
            for j in range(frame_width/2):
                for c in range(3):
                    new_frame[i][j][c] = int(frame[i+ height_offset][j + width_offset][c])


        output.append(new_frame)
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

output = np.asarray(output)
out_dict = {'frame_crop':output}
io.savemat(VIDEO_FILE + '_output.mat', out_dict)

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
