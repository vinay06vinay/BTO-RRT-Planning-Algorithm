import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from scipy.spatial.distance import pdist

'''
Function to Check Whether the given point is in obstacle space. Will return true if the obstacle is not in  Obstacle Space indicating 
clear path.
'''
def isObstacle(binary_image,node):
    x,y = node
    height,width = binary_image.shape
    if not ((round(x) >0 and round(x) < width) and (round(y) >0 and round(y) < height) and (binary_image[int(y),int(x)] != 0)):
        return False
    else:
        return True
def get_start_goal_inputs(binary_image):
    height,width = binary_image.shape
    while True:
        print("Enter the x-coordinate of start node")
        start_x = int(input())
        print("Enter the y-coordinate of start node")
        start_y = int(input())
        if((start_x <= 0 or start_x > width -1 ) ):
            print("The X-coordinate of start node is out of the map. Please enter the coordinates again")
        elif((start_y <= 0 or start_y >height - 1)):
            print("The Y-coordinate of start node is out of the map. Please enter the coordinates again")
        elif not (isObstacle(binary_image,(start_x,start_y))):
            print("The entered start node falls in obstacle space")
        else:
            break
    while True:
        print("Enter the x-coordinate of goal node")
        goal_x = int(input())
        print("Enter the y-coordinate of goal node")
        goal_y = int(input())
        if(goal_x <= 0 or goal_x >width -1):
            print("The X-coordinate of start node is out of the map. Please enter the coordinates again")
        elif(goal_y <= 0 and goal_y > height - 1):
            print("The Y-coordinate of start node is out of the map. Please enter the coordinates again")
        elif not(isObstacle(binary_image,(goal_x,goal_y))):
            print("The entered goal node falls in obstacle space")
        else:
            break
    return (start_x,start_y,goal_x,goal_y)
'''
Check Path checks whether the path between generated closest node and random new node doesnot fall in obstacle
space. This checks for different pose orientations on the new node.
'''
def checkpath(q_near, q_new, binary_image):
    check = True
    theta = math.atan2(q_new[1] - q_near[1], q_new[0] - q_near[0])
    i = 1
    while (i< int(math.sqrt(sum((q_near - q_new)**2))) + 1):
        poscheck = q_near + (i * np.array([math.cos(theta), math.sin(theta)]))
        poscheck_x,poscheck_y = poscheck[0],poscheck[1]
        if not (isObstacle(binary_image,(math.ceil(poscheck_x), math.ceil(poscheck_y))) and 
                isObstacle(binary_image,(math.floor(poscheck_x), math.floor(poscheck_y))) and
                isObstacle(binary_image,(math.ceil(poscheck_x), math.floor(poscheck_y))) and
                isObstacle(binary_image,(math.floor(poscheck_x), math.ceil(poscheck_y)))):
            check = False
            break
        q_new_x,q_new_y = q_new[0],q_new[1] 
        if not isObstacle(binary_image,(q_new_x,q_new_y)):
            check = False
            break
        i += 1
    return check
'''
The tree is extended by generating the sample target either by sampling random point or considering goal point as the sample such that more
pull is towards the goal either from start point or goal point.

Closest node in the tree and random point as in general RRT are considered and the euclidean distance is calculated. Finally, the new point which is under distance
threshold is considered the new point.
'''
def extend_rrt(rrt_tree_1, rrt_tree_2, goal_x,goal_y, step_size, maxFailedAttempts, distance_thresh, binary_image):
    rrt_tree_1 = np.reshape(rrt_tree_1,(-1,3))
    rrt_tree_2 = np.reshape(rrt_tree_2,(-1,3))
    path_found = []
    failedAttempts = 0
    extend_fail = True
    while failedAttempts <= maxFailedAttempts:
        if np.random.rand() < 0.15:
            x,y = (np.random.rand(1, 2) * np.array(binary_image.shape))[0][:]
            sample = np.array([x,y])
        else:
            #The goal x and goal y changes when we are extending tree from both Ends. For example,
            #When extending from start point , the goal will be towards tree on the goal side.
            #When extending from goal point, the goal will be on tree towards start side.
            sample = np.array([goal_x,goal_y])
        #For Each Point in the tree distance is calculated from the generated sample point above. 
        #Then a minimum distance is taken whose index gives the closest one to the current tree node
       
        distance_list = np.linalg.norm (rrt_tree_1[:, 0:2] - sample,axis = 1)#np.sqrt(np.sum((rrt_tree_1[:, 0:2] - sample)**2, axis=1))
        closest_node_index = np.argmin(distance_list)
        minimum_node = rrt_tree_1[closest_node_index, :]
        #Angle between the sample point and closest node is selected
        theta = math.atan2((sample[1]-minimum_node[1]),(sample[0]-minimum_node[0]))
        new_point = np.array([(minimum_node[0]+ step_size*(math.cos(theta))), (minimum_node[1]+ step_size*(math.sin(theta)))],dtype = np.float32)
        if checkpath(minimum_node[0:2],new_point, binary_image) == False:
            failedAttempts +=1
            continue
        distance1_list = np.linalg.norm (rrt_tree_2[:, 0:2] - new_point,axis = 1) 
        closest_node_index_1 = np.argmin(distance1_list)
        point_to_check_1 = rrt_tree_2[closest_node_index_1,0:2]
        distance_from_point_new_point_1 = np.linalg.norm(new_point - point_to_check_1)
        if distance1_list[closest_node_index_1] < distance_thresh: 
            node_x , node_y = new_point[0],new_point[1]
            path_found = [ node_x , node_y, closest_node_index, closest_node_index_1]
            extend_fail = False
            break
        distance2_list = np.linalg.norm (rrt_tree_1[:, 0:2]- new_point,axis = 1) 
        closest_node_index_2 = np.argmin(distance2_list)
        point_to_check = rrt_tree_1[closest_node_index_2,0:2]
        distance_from_point_new_point = np.linalg.norm(new_point - point_to_check) 
        if distance_from_point_new_point  < distance_thresh: 
            failedAttempts += 1
            continue
        rrt_tree_1 = np.vstack((rrt_tree_1, np.hstack((new_point, closest_node_index))))
        extend_fail = False
        break
    return rrt_tree_1, path_found, extend_fail
'''
Both trees from start and goal point start expanding until a path is found.
'''
def main_comb(binary_image, start_x,start_y, goal_x,goal_y, step_size, distance_thresh, maxFailedAttempts):
    rrt_tree_1 = np.array([start_x,start_y, -1], dtype=float)
    rrt_tree_2 = np.array([goal_x,goal_y, -1], dtype=float)
    expansion_state_1 = False
    expansion_state_2 = False
    tree1_p_list = []
    tree1_c_list = []
    tree2_p_list = []
    tree2_c_list = []
    
    while ((not expansion_state_1) or (not expansion_state_2)):
        if not expansion_state_1:
            rrt_tree_1, Tree_State, expansion_state_1 = extend_rrt(rrt_tree_1, rrt_tree_2, goal_x,goal_y, step_size, maxFailedAttempts, distance_thresh, binary_image)
            if not expansion_state_1 and len(Tree_State) == 0:
                parent_x,parent_y =  rrt_tree_1[int(rrt_tree_1[-1, 2]), 0] , rrt_tree_1[int(rrt_tree_1[-1, 2]), 1]
                child_x,child_y = rrt_tree_1[-1, 0] , rrt_tree_1[-1, 1]
                tree1_p_list.append((parent_x,parent_y))
                tree1_c_list.append((child_x,child_y))
                
        if not expansion_state_2:
            rrt_tree_2, Tree_State, expansion_state_2 = extend_rrt(rrt_tree_2, rrt_tree_1, start_x,start_y, step_size, maxFailedAttempts, distance_thresh, binary_image)
            if len(Tree_State) > 0:
                Tree_State[2:4] = Tree_State[2:4][::-1] 
            if not expansion_state_2 and len(Tree_State)  == 0:
                parent_x,parent_y =  rrt_tree_2[int(rrt_tree_2[-1, 2]), 0] , rrt_tree_2[int(rrt_tree_2[-1, 2]), 1]
                child_x,child_y = rrt_tree_2[-1, 0] , rrt_tree_2[-1, 1]
                tree2_p_list.append((parent_x,parent_y))
                tree2_c_list.append((child_x,child_y))
        final_line = []       
        if len(Tree_State)  > 0:
            ptA = (rrt_tree_1[Tree_State[2], 0],rrt_tree_1[Tree_State[2], 1])
            ptB = (rrt_tree_2[Tree_State[3], 0],rrt_tree_2[Tree_State[3], 1])
            pt_join = (Tree_State[0],Tree_State[1])
            final_line.append(ptA)
            final_line.append(ptB)
            final_line.append(pt_join)
            path = Tree_State[ 0:2]
            prev = Tree_State[ 2]
            while prev > 0:
                path = np.vstack((rrt_tree_1[int(prev), 0:2], path))
                prev = rrt_tree_1[int(prev), 2]
            prev = Tree_State[ 3]
            while prev > 0:
                path = np.vstack((path, rrt_tree_2[int(prev), 0:2]))
                prev = rrt_tree_2[int(prev), 2]
            break
    if len(Tree_State) == 0:
        raise ValueError('no path found. maximum attempts reached')
    return path,final_line, tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list

    
def down_path_sampling(binary_image, path,goal_x,goal_y):
    k = 0
    path_length = path.shape[0]
    down_path = path[0]
    temp_node = path[0]
    distances = np.zeros(path_length)
    for i in range(1, path_length):
        current_node = path[i] 
        distances[i] = np.linalg.norm(current_node - temp_node)
        if distances[i] > 0 or distances[i] > distances[i-1]:
            current_index = i
            new_node =  path[current_index] 
        else:
            new_node = np.array([])
        if len(new_node) != 0:
            if not checkpath(temp_node, new_node, binary_image):
                temp_node = np.array([path[current_index-1, 0], path[current_index-1, 1]])
                down_path = np.vstack([down_path, temp_node])
    goal = np.array([goal_x,goal_y])
    down_path = np.vstack([down_path, goal])
    return down_path

def visualise(image,final_line,path,tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list,down_path,up_path,image_copy,up_path_1000,up_path_10000):
    image_copy_1000 = image_copy.copy()
    image_copy_10000 = image_copy.copy()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    
    out = cv2.VideoWriter('BTO-RRT.avi',fourcc,250,(image.shape[1],image.shape[0]))
    for parent1,child1,parent2,child2 in zip(tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list):
        parent1_x,parent1_y = round(parent1[0]),round(parent1[1])
        child1_x,child1_y = round(child1[0]),round(child1[1])
        parent2_x,parent2_y = round(parent2[0]),round(parent2[1])
        child2_x,child2_y = round(child2[0]),round(child2[1])
        cv2.line(image,(parent1_x,parent1_y),(child1_x,child1_y),(255,0,0), 1,cv2.LINE_AA)
        cv2.line(image,(parent2_x,parent2_y),(child2_x,child2_y),(0,0,255), 1,cv2.LINE_AA)
        for i in range(30):
            out.write(image)
            cv2.waitKey(1)
    ptA_final_x , ptA_final_y = round(final_line[0][0]),round(final_line[0][1])
    ptB_final_x , ptB_final_y = round(final_line[1][0]),round(final_line[1][1])
    common_x , common_y = round(final_line[2][0]),round(final_line[2][1])
    cv2.line(image,(ptA_final_x , ptA_final_y),(common_x , common_y),(0,0,0), 2,cv2.LINE_AA)
    cv2.line(image,(ptB_final_x , ptB_final_y),(common_x , common_y),(0,0,0), 2,cv2.LINE_AA)
    out.write(image)
    cv2.imwrite("BTO-RRT-explore.jpg",image)
    cv2.waitKey(1)
    for i in range(len(path) - 1):
        ptA = (round(path[i][0]),round(path[i][1]))
        ptB = (round(path[i+1][0]),round(path[i+1][1]))
        cv2.line(image,ptA,ptB,(0,255,0), 2,cv2.LINE_AA)
        cv2.line(image_copy,ptA,ptB,(0,255,0), 1,cv2.LINE_AA)
        cv2.line(image_copy_1000,ptA,ptB,(0,255,0), 1,cv2.LINE_AA)
        cv2.line(image_copy_10000,ptA,ptB,(0,255,0), 1,cv2.LINE_AA)
        for i in range(20):
            out.write(image)
            cv2.waitKey(1)
    cv2.imwrite("BTO-RRT.jpg",image)
    for i in range(len(down_path) - 1):
        ptA = (round(down_path[i][0]),round(down_path[i][1]))
        ptB = (round(down_path[i+1][0]),round(down_path[i+1][1]))
        cv2.line(image,ptA,ptB,(255,255,0), 1,cv2.LINE_AA)
        cv2.line(image_copy,ptA,ptB,(255,255,0), 1,cv2.LINE_AA)
        cv2.line(image_copy_1000,ptA,ptB,(255,255,0), 1,cv2.LINE_AA)
        cv2.line(image_copy_10000,ptA,ptB,(255,255,0), 1,cv2.LINE_AA)
        for i in range(100):
            out.write(image)
            cv2.waitKey(1)
    cv2.imwrite("BTO-RRT-DownSample.jpg",image)
    for i in range(len(up_path) - 1):
        ptA = (round(up_path[i][0]),round(up_path[i][1]))
        ptB = (round(up_path[i+1][0]),round(up_path[i+1][1]))
        cv2.line(image,ptA,ptB,(151,180,184), 2,cv2.LINE_AA)
        cv2.line(image_copy,ptA,ptB,(151,101,184), 1,cv2.LINE_AA)
        for i in range(20):
            out.write(image)
            cv2.waitKey(1)
    for i in range(len(up_path_1000) - 1):
        ptA = (round(up_path_1000[i][0]),round(up_path_1000[i][1]))
        ptB = (round(up_path_1000[i+1][0]),round(up_path_1000[i+1][1]))
        cv2.line(image_copy_1000,ptA,ptB,(179,119,15), 1,cv2.LINE_AA)
    for i in range(len(up_path_10000) - 1):
        ptA = (round(up_path_10000[i][0]),round(up_path_10000[i][1]))
        ptB = (round(up_path_10000[i+1][0]),round(up_path_10000[i+1][1]))
        cv2.line(image_copy_10000,ptA,ptB,(179,15,64), 1,cv2.LINE_AA)

    cv2.imwrite("BTO-RRT-UpSample.jpg",image)
    cv2.imwrite("BTO-RRT-Comparsion.jpg",image_copy)
    cv2.imwrite("BTO-RRT-Comparsion-1000.jpg",image_copy_1000)
    cv2.imwrite("BTO-RRT-Comparsion-10000.jpg",image_copy_10000)
    out.release()
    cv2.destroyAllWindows()
    return image  

def up_sample_path(down_path,image):
    P = np.transpose(down_path)
    n, m = np.shape(P)
    l = np.zeros((m,1))
    for k in range(1, m):
        l[k] = np.linalg.norm(P[:,k]-P[:,k-1]) + l[k-1]
    l_init = l[m-1]
    current_iteration = 1
    max_iteration_list = [500,1000,10000]
    up_path_list = []
    for count in range(3):
        while current_iteration <= max_iteration_list[count]:
            random_node1 = np.random.rand(1)*l[m-1]
            random_node2 = np.random.rand(1)*l[m-1]
            if random_node2 < random_node1:
                random_node1,random_node2 = random_node2,random_node1
            for k in range(1, m):
                if random_node1 < l[k]:
                    i = k-1
                    break
            for k in range(i+1, m):
                if random_node2 < l[k]:
                    j = k-1
                    break
            if j <= i:
                current_iteration += 1
                continue
            t1 = (random_node1 - l[i])/(l[i+1] - l[i])
            gamma_1 = (1-t1)*P[:,i] + t1*P[:,i+1]
            t2 = (random_node2 - l[j])/(l[j+1] - l[j])
            gamma_2 = (1-t2)*P[:,j] + t2*P[:,j+1]
            if not checkpath(np.round(gamma_1), np.round(gamma_2), image):
                current_iteration += 1
                continue
            new_point = np.concatenate((P[:,0:i+1], np.round(gamma_1).reshape((-1,1)), np.round(gamma_2).reshape((-1,1)), P[:,j+1:m]), axis=1)
            P = new_point
            n, m = np.shape(P)
            l = np.zeros((m,1))
            for k in range(1, m):
                l[k] = np.linalg.norm(P[:,k]-P[:,k-1]) + l[k-1]
            current_iteration += 1
        up_path = np.transpose(P)    
        up_path_list.append(np.transpose(P))
    return up_path_list
'''
Calculating the path length
'''
def calculate_distance(path):
    distance_list = []
    for i in range(len(path)-1):
        distance = np.linalg.norm (path[i+1] - path[i])
        distance_list.append(distance)
    return sum(distance_list)
def main():
    image =  cv2.imread("map7.bmp")
    image_copy = image.copy()
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,binary_image = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    start_x,start_y,goal_x,goal_y = get_start_goal_inputs(binary_image)

    start = (start_x,start_y)
    goal = (goal_x,goal_y)
    print("The Start Node selected is :",start)
    print("The Goal Node Selected is :", goal)
    #Variables
    distance_thresh = 10
    step_size = 20
    maxFailedAttempts = 10000
    start = time.time()
    path,final_line, tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list = main_comb(binary_image, start_x,start_y, goal_x,goal_y, step_size, distance_thresh, maxFailedAttempts)
    end = time.time()
    
    print("The BTO-RRT final path :",path)
    print("The Time Taken for BTO RRT exploration:", end-start ,"seconds")
    down_path = down_path_sampling(binary_image, path ,goal_x,goal_y)
    print("Optimized Path after DownSample:",down_path)
    up_path,up_path_1000,up_path_10000 = up_sample_path(down_path,binary_image)
    print("Optimized Path after UpSample:",up_path)
    image = visualise(image,final_line,path,tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list,down_path,up_path,image_copy,up_path_1000,up_path_10000)
    path_length_bto = calculate_distance(path)
    down_sample_length = calculate_distance(down_path)
    up_sample_length = calculate_distance(up_path)
    up_sample_length_1000 = calculate_distance(up_path_1000)
    up_sample_length_10000 = calculate_distance(up_path_10000)
    print(len(up_path),len(up_path_1000),len(up_path_10000))
    print("The Total Path Length for BTO-RRT:",path_length_bto )
    print("The Total Path Length for BTO-RRT-DownSample:", down_sample_length)
    print("The Total Path Length for BTO-RRT-UpSample for 500 iterations:",up_sample_length )
    print("The Total Path Length for BTO-RRT-UpSample for 1000 iterations:",up_sample_length_1000 )
    print("The Total Path Length for BTO-RRT-UpSample for 10000 iterations:",up_sample_length_10000 )
            
    
    
if __name__ == '__main__':
    main()