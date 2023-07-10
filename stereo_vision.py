import numpy as np
import cv2
from pipeline import pipeline

# Take choice of input images from user
while True:
    print("Select one of the following options:")
    print("1. Curule")
    print("2. Octagon")
    print("3. Pendulum")

    try:
        choice=int(input('Your choice: '))
        if choice not in range(1,4):
            print("Invalid choice, please try again")
            continue
        break
    except:
        print("Not a number, please try again")
        continue 

if choice==1:
    folder='curule'
elif choice==2:
    folder='octagon'
else:
    folder='pendulum'

# Read Images
img1=cv2.imread(folder+'/im0.png')
img2=cv2.imread(folder+'/im1.png')
# cv2.imshow('Image 0',img1)
# cv2.waitKey(0)
# cv2.imshow('Image 1', img2)
# cv2.waitKey(0)

# Open calibration file
fh=open(folder+'/calib.txt')

# Obtain calibration parameters from calibration file
for line in fh:
    line=line.lstrip().rstrip().rstrip(']')
    sides=line.split('=')
    right_ele=sides[1].split(';')
    if len(right_ele)==3:
        temp_array=[]
        for i in right_ele:
            i=i.lstrip().lstrip('[').rstrip().rstrip(']')
            i=i.split()
            for ele in i:
                temp_array.append(float(ele))
        cam_param=np.array(list(temp_array))
        cam_param=np.reshape(cam_param, (3,3))
        locals()[sides[0]]=np.array(cam_param)
    else:
        locals()[sides[0]]=float(sides[1])

focal_length=cam0[0][0]

# print(cam0,cam0.shape)
# print(cam1,cam1.shape)
# print(doffs)
# print(baseline)
# print(width)
# print(height)
# print(ndisp)
# print(vmin)
# print(vmax)
# print(focal_length)

# Pipeline called
pipeline(img1,img2,cam0,cam1,doffs,baseline,width,height,ndisp,vmin,vmax,focal_length)