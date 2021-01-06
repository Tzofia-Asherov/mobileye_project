import numpy as np
import math

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_curr_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container

def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)

    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ

def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
 
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normalize_points = []
    for point in pts:
        x = (point[0] - pp[0]) / focal
        y = (point[1] - pp[1]) / focal
        normalize_points.append([x,y])
    return np.array([np.array(point_list) for point_list in normalize_points])


    
def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    un_normalize_points = []
    for point in pts:
        x = (point[0]* focal) + pp[0]
        y = (point[1] * focal) + pp[1]
        un_normalize_points.append([x,y])
    return np.array([np.array(point_list) for point_list in un_normalize_points])

    

def decompose(EM):
    R = EM[0:3,0:3]
    tZ = EM[0:3,3]
    feo = [tZ[0] / tZ[2] , tZ[1] / tZ[2]]
    return R, feo, tZ[2]

def rotate(pts, R):
    # rotate the points - pts using R
    ret_point=[]
    for point in pts:
        arr=R.dot([point[0],point[1],1])
        ret_point.append((arr[0],arr[1])/arr[2])
    return ret_point

def find_corresponding_points(p, norm_pts_rot, foe):
    m=(foe[1]-p[1])/(foe[0]-p[0])
    n=(p[1]*foe[0]-p[0]*foe[1])/(foe[0]-p[0])
    ret= min([(i,norm_pts_rot[i][0],norm_pts_rot[i][1]) for i in range(len(norm_pts_rot))] ,key=lambda t: distance(m,n,t[1],t[2]))

    return ret[0],[ret[1],ret[2]]
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index

def distance(m,n,x,y):
    return abs((m*x+n-y)/ math.sqrt(m*m+1))

def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    Z_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])  
    Z_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
   
    dis_x=abs(p_curr[0]-foe[0])
    dis_y=abs(p_curr[1]-foe[1])
    ret = (abs(Z_x *dis_x ) + abs(Z_y * dis_y)) / abs(dis_x+dis_y)
    
    return ret