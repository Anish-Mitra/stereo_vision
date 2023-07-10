import matplotlib.pyplot as plt
import numpy as np
import cv2
from calibration import feature_matcher, F_Ransac_Matrix, Essential_matrix, camera_pose,points_triangulation, correct_camerapose
from rectification import rectification, draw_epilines
from correspondence import ssd_correspondence
from compute_depth import disparity_to_depth

# Function to create the pipeline for stereo vision by calling all necessary pipeline element functions
def pipeline(img1,img2,cam0, cam1, doffs, baseline, width, height, ndisp, vmin, vmax, focal_length):

    
    ################################# Calibration phase
    matched_pairs=feature_matcher(img1, img2)
    F,sift_inliers=F_Ransac_Matrix(matched_pairs)
    E=Essential_matrix(F, cam0, cam1)
    print('Essential Matrix E: ',E)
    rotational,translational=camera_pose(E)
    triangulated_pts=points_triangulation(cam0, cam1, rotational, translational, sift_inliers)
    count_1 = []
    count_2 = []
    rot = np.identity(3)
    tran = np.zeros((3, 1))
    for i in range(len(triangulated_pts)):
        points3D = triangulated_pts[i]
        points3D = points3D/points3D[3, :]
        # Get only positive z values
        count_2.append(correct_camerapose(points3D, rotational[i], translational[i]))
        count_1.append(correct_camerapose(points3D, rot, tran))
    count_1 = np.array(count_1)
    count_2 = np.array(count_2)
    print("count1 count2 ",count_1,count_2)
    count_threshold = int(triangulated_pts[0].shape[1] / 2)
    idx = np.intersect1d(np.where(count_1 > count_threshold),np.where(count_2 > count_threshold))
    
    best_rot_matrix = rotational[idx[0]]
    best_trans_matrix = translational[idx[0]]
    
    print("Optimal Camera poses: ")
    print("Rotation Matrix ", best_rot_matrix)
    print("Translational Matrix ",best_trans_matrix)
    

    ################################# Rectification Phase
    pts1,pts2=sift_inliers[:,0:2],sift_inliers[:,2:4]
    # print('points1,points2 ',pts1,pts2)
    rectified_pts1, rectified_pts2, rectified_img1, rectified_img2=rectification(img1, img2, pts1, pts2, F)
          
    
    # ################################ Correspondence Phase
    disparity_map_scaled=ssd_correspondence(rectified_img1, rectified_img2)
    plt.figure(1)
    plt.title('Disparity Map Graysacle')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.figure(2)
    plt.title('Disparity Map Hot')
    plt.imshow(disparity_map_scaled, cmap='hot')
    plt.show()

    # ############################## Depth Map Calculation
    depth_map = disparity_to_depth(baseline, focal_length, disparity_map_scaled)

    plt.figure(3)
    plt.title('Depth Map Graysacle')
    plt.imshow(depth_map, cmap='gray')
    plt.figure(4)
    plt.title('Depth Map Hot')
    plt.imshow(depth_map, cmap='hot')
    plt.show()


