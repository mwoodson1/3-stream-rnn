import cv2
import numpy as np
import scipy.io as sio

def foveaCrop(fileName, Dir, vidName)
    output = []
    vidName = 'UCF-101/BasketballDunk/v_BasketballDunk_g01_c01.avi'
    #vidName = 'drop.avi'
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(vidName)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(vidName,fourcc, 20.0, (640,480))

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
    sio.savemat(directory + '/' + vidname.split(".")[0]+"_fcrop.mat", out_dict)

    # Release everything if job is finished
    cap.release()




if __name__ == "__main__":

    dirs = [x[0] for x in os.walk("../data/UCF-101/")]

    for i in xrange(len(dirs)):
        if(i==0):
            continue
        directory = "../data/pre-process/"+dirs[i].split("/")[3]
        if not os.path.exists(directory):
            os.makedirs(directory)

    #Loop through all directories
    for i in xrange(len(dirs)):
        if(i==0):
            continue
        #Loop through every file in the directory
        for filename in os.listdir(dirs[i]):
            make_optFlow(dirs[i]+"/"+filename,dirs[i],filename)



