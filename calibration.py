import numpy as np
import cv2
import math 
import random as rd

def normalize(uv):
    
    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

# Function to detect matching features in two images
def feature_matcher(img1,img2):

    img1_sift=img1.copy()
    img2_sift=img2.copy()
    img1_sift_gray = cv2.cvtColor(img1_sift, cv2.COLOR_BGR2GRAY)
    img2_sift_gray = cv2.cvtColor(img2_sift, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    # Find keypoints in each image
    keypoints1, descriptors1 = sift.detectAndCompute(img1_sift_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_sift_gray, None)

    # create Brute Force Matcher object
    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    
    matches=matches[0:100]
    # Visualize matched keypoints
    matched_keypoints=cv2.drawMatches(img1_sift_gray, keypoints1, img2_sift_gray, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite('Matched_keypoints_pendulum.png',matched_keypoints)
    # cv2.waitKey(0) 

    # Extract matched x,y coordinates
    matched_pairs=[]
    for i,m in enumerate(matches):
        points1=keypoints1[m.queryIdx].pt
        points2=keypoints2[m.trainIdx].pt
        matched_pairs.append([points1[0], points1[1], points2[0], points2[1]])
    matched_pairs=np.array(matched_pairs).reshape(-1,4)
    # kp1_matched_list = [list(keypoints1[mat.queryIdx].pt) for mat in matches]
    # kp2_matched_list = [list(keypoints2[mat.trainIdx].pt) for mat in matches]
    # matched_pairs=np.array([kp1_matched_list,kp2_matched_list]).reshape(-1,4)
    print('matched pairs shape',matched_pairs.shape)
    return matched_pairs

# Function to find fundamental matrix
def fundamental_matrix(matched_pairs):
    x1 = matched_pairs[:, 0:2]
    x2 = matched_pairs[:, 2:4]
    if(x1.shape[0] > 7):
        x1_norm, T1 = normalize(x1)
        x2_norm, T2 = normalize(x2)
        A = np.zeros((len(x1_norm), 9))
        for i in range(0, len(x1_norm)):
            x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2, y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2 *x_1, y_2*y_1, y_2, x_1, y_1, 1])

        # Applying SVD to find F matrix
        [U,Sigma,Vt]=np.linalg.svd(A,full_matrices=True)
        F=Vt.T[:,-1]
        F=F.reshape(3,3)

        # Reduce rank of F to make epipolar lines meet
        u, sigma, vt = np.linalg.svd(F)
        sigma = np.diag(sigma)
        sigma[2, 2] = 0.0
        F = np.dot(u, np.dot(sigma, vt))
        F = np.dot(T2.T, np.dot(F, T1))
        return F
    else:
        return None


def fundamental_error(feature, F):
    x1, x2 = feature[0:2], feature[2:4]
    x1_temp = np.transpose(np.array([x1[0], x1[1], 1]))
    x2_temp = np.array([x2[0], x2[1], 1])
    error = np.dot(x1_temp, np.dot(F, x2_temp))
    return np.abs(error)

# Function to find best fundamental matrix using RANSAC 
def F_Ransac_Matrix(features):
    
    err_thresh=0.02
    inliers_thresh=0
    Best_F_matrix=[]
    for i in range(0, 1000):
        indices = []
        n_rows = features.shape[0]
        random_pairs = np.random.choice(n_rows, size=8)
        eight_features = features[random_pairs, :]
        F = fundamental_matrix(eight_features)
        for j in range(n_rows):
            feature = features[j]
            error = fundamental_error(feature, F)
            if (error < err_thresh):
                indices.append(j)
        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            selected_indices = indices
            Best_F_matrix = F
    filtered_features = features[selected_indices, :]
    print("Best Fundamental matrix ", Best_F_matrix)
    return Best_F_matrix,filtered_features

# Function to find essential matrix
def Essential_matrix(F, cam0, cam1):
    
    E = np.dot(np.transpose(cam1), np.dot(F, cam0))
    U_e, Sigma_e, Ve_t = np.linalg.svd(E)
    Sigma_e = [1, 1, 0]
    final_E = np.dot(U_e, np.dot(np.diag(Sigma_e), Ve_t))
    return final_E
    

# Function to derive camera pose
def camera_pose(E):

    U_e,sigma_e,Ve_t=np.linalg.svd(E)
    d = np.array([[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]])

    temp1_c, temp2_c = U_e[:, 2], -U_e[:, 2]
    temp1_r, temp2_r = np.dot(U_e, np.dot(d, Ve_t)), np.dot(U_e, np.dot(np.transpose(d), Ve_t))

    # Rotational matrices
    rotational=[temp1_r,temp1_r,temp2_r,temp2_r]
    # Translational matrices
    translational=[temp1_c,temp2_c,temp1_c,temp2_c]
    for i in range(4):
        if(np.linalg.det(rotational[i]) < 0):
            rotational[i] = -rotational[i]
            translational[i] = -translational[i]
    return rotational, translational
    
# Function to find traiangulation points based on camera parameters and cheirality conditions
def points_triangulation(cam0, cam1, rotational,translational,inliers):
    triangulated_pts = []
    rotational_1 = np.identity(3)
    translational_1 = np.zeros((3, 1))
    I = np.identity(3)
    P1 = np.dot(cam0, np.dot(rotational_1, np.hstack((I, -translational_1.reshape(3, 1)))))
    for i in range(len(translational)):
        x1 = np.transpose(inliers[:, 0:2])
        x2 = np.transpose(inliers[:, 2:4])
        P2 = np.dot(cam1, np.dot(rotational_1, np.hstack((I, -translational[i].reshape(3, 1)))))
        X = cv2.triangulatePoints(P1, P2, x1, x2)
        triangulated_pts.append(X)
    return triangulated_pts

def correct_camerapose(pts,rot,trans):
    
    I = np.identity(3)
    P = np.dot(rot, np.hstack((I, -trans.reshape(3, 1))))
    P = np.vstack((P, np.array([0, 0, 0, 1]).reshape(1, 4)))
    n_positive = 0
    for i in range(pts.shape[1]):
        X = pts[:, i]
        X = X.reshape(4, 1)
        X_c = np.dot(P, X)
        X_c = X_c/X_c[3]
        z = X_c[2]
        if(z > 0):
            n_positive += 1
    return n_positive



