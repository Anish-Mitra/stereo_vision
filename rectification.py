import numpy as np
import cv2

# Function to match the sizes of the image
def match_image_sizes(image_list):
    images = image_list.copy()
    sizes = []
    for image in images:
        x, y, channel = image.shape
        sizes.append([x, y, channel])
    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis=0)
    resized_images = []
    for i, image in enumerate(images):
        resized_image = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        resized_image[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        resized_images.append(resized_image)
    return resized_images

def compute_x(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

# Function to draw the epipolar lines
def draw_epilines(img1,img2,pts1,pts2,F,rectified=False):

    epilines_1, epilines_2 = [], []
    img_1 = img1.copy()
    img_2 = img2.copy()
    for i in range(pts1.shape[0]):
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1]).reshape(3, 1)
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1]).reshape(3, 1)
        line_2 = np.dot(F, x1)
        epilines_2.append(line_2)
        line_1 = np.dot(F.T, x2)
        epilines_1.append(line_1)
        if not rectified:
            y2_min = 0
            y2_max = img2.shape[0]
            x2_min = compute_x(line_2, y2_min)
            x2_max = compute_x(line_2, y2_max)
            y1_min = 0
            y1_max = img1.shape[0]
            x1_min = compute_x(line_1, y1_min)
            x1_max = compute_x(line_1, y1_max)
        else:
            x2_min = 0
            x2_max = img2.shape[1] - 1
            y2_min = -line_2[2]/line_2[1]
            y2_max = -line_2[2]/line_2[1]
            x1_min = 0
            x1_max = img1.shape[1] - 1
            y1_min = -line_1[2]/line_1[1]
            y1_max = -line_1[2]/line_1[1]
        cv2.circle(img_2, (int(pts2[i, 0]), int(pts2[i, 1])), 10, (0, 0, 255), -1)
        img_2 = cv2.line(img_2, (int(x2_min), int(y2_min)),(int(x2_max), int(y2_max)), (int(i*2.55), 255, 0), 2)
        cv2.circle(img_1, (int(pts1[i, 0]), int(pts1[i, 1])), 10, (0, 0, 255), -1)
        img_1 = cv2.line(img_1, (int(x1_min), int(y1_min)),(int(x1_max), int(y1_max)), (int(i*2.55), 255, 0), 2)
    image_1, image_2 = match_image_sizes([img_1, img_2])
    final_img = np.concatenate((image_1, image_2), axis=1)
    final_img = cv2.resize(final_img, (1920, 660))
    # cv2.imshow('Left',image_1)
    # cv2.imshow('Right', image_2)
    # cv2.waitKey(0)
    return epilines_1, epilines_2, final_img

# Function to perform rectification phase in pipeline
def rectification(img1, img2, pts1, pts2, F):

    pre_epilines1,pre_epilines2,unrectified_output=draw_epilines(img1, img2, pts1, pts2, F,False)
    cv2.imwrite('unrectified_epilines_pendulum.png',unrectified_output)
    # cv2.waitKey(0)

    # Stereo rectification
    ht1, wd1 = img1.shape[:2]
    ht2, wd2 = img2.shape[:2]

    # Find homography of both images when camera parameters are not given
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(wd1, ht1))
    print("H1", H1)
    print("H2", H2)
    
    rectified_img1 = cv2.warpPerspective(img1, H1, (wd1, ht1))
    rectified_img2 = cv2.warpPerspective(img2, H2, (wd2, ht2))
    rectified_pts1 = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H1).reshape(-1, 2)
    rectified_pts2 = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H2).reshape(-1, 2)
    
    H1_inv = np.linalg.inv(H1)
    H2_T_inv = np.linalg.inv(np.transpose(H2))
    rectified_F = np.dot(H2_T_inv, np.dot(F, H1_inv))
    rectified_epilines_1,rectified_epilines_2,rectified_output=draw_epilines(rectified_img1, rectified_img2, rectified_pts1, rectified_pts2, rectified_F,True)
    # cv2.imshow("rectified_1", rectified_img1)
    # cv2.imshow("rectified_2", rectified_img2)
    # cv2.imshow('rectified output', rectified_output)
    # cv2.waitKey(0)
    cv2.imwrite('rectified_epilines_pendulum.png', rectified_output)
    
        
    return rectified_pts1, rectified_pts2, rectified_img1, rectified_img2
