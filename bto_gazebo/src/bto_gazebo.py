import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import rospy
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

def obstacle_space(width,height,canvas,robot_radius,clearance):
    offset  = robot_radius+clearance
    for x in range(width):
        for y in range(height):
#           #Obstacle map and matrix formation with offset consideration
           # #First Rectangle
            r1c = (x+offset>=1500 and x-offset<=1650) and (y+offset>=0 and y-offset<=1250)
            #second Rectangle
            r2c = (x+offset>=2500 and x-offset<=2650) and (y+offset>= 750 and y-offset<=2000)
            #Circle with clearance. The offset will be added to the radius
            c1c = ((x-4000)**2 + (y-1100)**2 - ((500+offset)**2) <= 0)
            if(r1c or r2c or  c1c):
                canvas[y,x] = (0,0,0)
            else:
                canvas[y,x] = (255,255,255)
    return canvas
def isObstacle(binary_image,node):
    #true if in obstacle space or node region
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
        elif((start_y <= 0 or 2000-start_y >height - 1)):
            print("The Y-coordinate of start node is out of the map. Please enter the coordinates again")
        elif not (isObstacle(binary_image,(start_x,2000-start_y))):
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
        elif(goal_y <= 0 and 2000-goal_y > height - 1):
            print("The Y-coordinate of start node is out of the map. Please enter the coordinates again")
        elif not(isObstacle(binary_image,(goal_x,2000-goal_y))):
            print("The entered goal node falls in obstacle space")
        else:
            break
    return (start_x,start_y,goal_x,goal_y)
'''
Check Path checks whether the path between generated closest node and random new node doesnot fall in obstacle
space.
'''
def checkpath(q_near, q_new, binary_image):
    check = True
    theta = math.atan2(q_new[1] - q_near[1], q_new[0] - q_near[0])
    i = 1
    while (i< int(math.sqrt(sum((q_near - q_new)**2))) + 1):
        
        #print(int(math.sqrt(sum((q_near - q_new)**2))) + 1)
        # print(i,i * [math.cos(theta), math.sin(theta)], [math.cos(theta), math.sin(theta)])
        poscheck = q_near + (i * np.array([math.cos(theta), math.sin(theta)]))
        poscheck_x,poscheck_y = poscheck[0],poscheck[1]
        # if not (isObstacle(binary_image,(math.ceil(poscheck_x), math.ceil(poscheck_y)))):
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
def extend_rrt(rrt_tree_1, rrt_tree_2, goal_x,goal_y, step_size, maxFailedAttempts, distance_thresh, binary_image):
    rrt_tree_1 = np.reshape(rrt_tree_1,(-1,3))
    rrt_tree_2 = np.reshape(rrt_tree_2,(-1,3))
    path_found = []
    failedAttempts = 0
    extend_fail = True
    height,width = binary_image.shape
    while failedAttempts <= maxFailedAttempts:
        if np.random.rand() < 0.4:
            x,y = (np.random.rand(1, 2) * np.array([width,height]))[0][:]
            sample = np.array([x,y])
        else:
            #The goal x and goal y changes when we are extending tree from both Ends. For example,
            #When extending from start point , the goal will be towards tree on the goal side.
            #When extending from goal point, the goal will be on tree towards start side.
            sample = np.array([goal_x,goal_y])
        #For Each Point in the tree distance is calculated from the generated sample point above. 
        #Then a minimum distance is taken whose index gives the closest one to the current tree node
       
        distance_list = np.linalg.norm (rrt_tree_1[:, 0:2] - sample,axis = 1)
        closest_node_index = np.argmin(distance_list)
        minimum_node = rrt_tree_1[closest_node_index, :]
        #Angle between the sample point and closest node is selected
        theta = math.atan2((sample[1]-minimum_node[1]),(sample[0]-minimum_node[0]))
        new_point = np.array([(minimum_node[0]+ step_size*(math.cos(theta))), (minimum_node[1]+ step_size*(math.sin(theta)))],dtype = np.float32)
        if checkpath(minimum_node[0:2],new_point, binary_image) == False:
            failedAttempts +=1
            continue
        distance1_list = np.linalg.norm (rrt_tree_2[:, 0:2] - new_point,axis = 1) #np.sqrt(np.sum((rrt_tree_2[:, 0:2] - new_point)**2, axis=1))
        closest_node_index_1 = np.argmin(distance1_list)
        point_to_check_1 = rrt_tree_2[closest_node_index_1,0:2]
        distance_from_point_new_point_1 = np.linalg.norm(new_point - point_to_check_1)
        if distance1_list[closest_node_index_1] < distance_thresh: 
            # if both trees are connected
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
def visualise(image,final_line,path,tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list,down_path):
    for parent1,child1,parent2,child2 in zip(tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list):
        parent1_x,parent1_y = round(parent1[0]),round(parent1[1])
        child1_x,child1_y = round(child1[0]),round(child1[1])
        parent2_x,parent2_y = round(parent2[0]),round(parent2[1])
        child2_x,child2_y = round(child2[0]),round(child2[1])
        cv2.line(image,(parent1_x,parent1_y),(child1_x,child1_y),(255,0,0), 1,cv2.LINE_AA)
        cv2.line(image,(parent2_x,parent2_y),(child2_x,child2_y),(0,0,255), 1,cv2.LINE_AA)
    ptA_final_x , ptA_final_y = round(final_line[0][0]),round(final_line[0][1])
    ptB_final_x , ptB_final_y = round(final_line[1][0]),round(final_line[1][1])
    common_x , common_y = round(final_line[2][0]),round(final_line[2][1])
    cv2.line(image,(ptA_final_x , ptA_final_y),(common_x , common_y),(0,0,0), 2,cv2.LINE_AA)
    cv2.line(image,(ptB_final_x , ptB_final_y),(common_x , common_y),(0,0,0), 2,cv2.LINE_AA)
    cv2.imwrite("BTO-RRT-explore-gazebo.jpg",image)
    for i in range(len(path) - 1):
        ptA = (round(path[i][0]),round(path[i][1]))
        ptB = (round(path[i+1][0]),round(path[i+1][1]))
        cv2.line(image,ptA,ptB,(0,255,0), 2,cv2.LINE_AA)
    cv2.imwrite("BTO-RRT-gazebo.jpg",image)
    # out.release()
    for i in range(len(down_path) - 1):
        ptA = (round(down_path[i][0]),round(down_path[i][1]))
        ptB = (round(down_path[i+1][0]),round(down_path[i+1][1]))
        cv2.line(image,ptA,ptB,(255,255,0), 1,cv2.LINE_AA)
    cv2.imwrite("BTO-RRT-DownSample-Gazebo.jpg",image)
    cv2.destroyAllWindows()
    return image

    
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


'''For Gazebo Simulation
'''
def calculate_theta_gazebo(ptA,ptB):
    ptA_x,ptA_y = ptA[0],ptA[1]
    ptB_x,ptB_y = ptB[0],ptB[1]
    angle = math.atan2((ptB_y-ptA_y),(ptB_x-ptA_x))
    return angle
def calculate_distance_gazebo(ptA,ptB):
    distance = np.linalg.norm (ptB - ptA)
    return distance
def position_callback(pose_message):
    global pose_x
    global pose_y
    global angle_z
    pose_x = pose_message.pose.pose.position.x
    pose_y = pose_message.pose.pose.position.y
    quaternion = (
        pose_message.pose.pose.orientation.x,
        pose_message.pose.pose.orientation.y,
        pose_message.pose.pose.orientation.z,
        pose_message.pose.pose.orientation.w
    )
    _, _, angle_z = euler_from_quaternion(quaternion)

def go_to_goal(angle,distance_i,goal,vel_value,publisher,r):
    print("The Goal",goal)
    global pose_x
    global pose_y
    global angle_z
    k_linear = 0.15
    k_angular = 0.09
    while True:
        current_node = np.array([pose_x,pose_y])
        distance = calculate_distance_gazebo(current_node,goal)
        linear_speed = k_linear * distance
        desired_angle_goal = calculate_theta_gazebo(current_node,goal)
        angle_speed = (desired_angle_goal-angle_z)*k_angular
        vel_value.linear.x = linear_speed
        vel_value.angular.z = angle_speed
        publisher.publish(vel_value)
        # print(distance,goal,current_node)
        if(distance < 0.02):
            break
        r.sleep()
    
def ros_visualisation(down_path):
    global pose_x
    global pose_y
    global angle_z
    rospy.init_node('bto_gazebo',anonymous=True)
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    r = rospy.Rate(30)
    rospy.sleep(1)
    pose_subscriber = rospy.Subscriber('/odom', Odometry, position_callback) 

    vel_value =  Twist()
    vel_value.linear.x = 0.0
    vel_value.linear.y = 0.0
    vel_value.linear.z = 0.0
    vel_value.angular.x = 0.0
    vel_value.angular.y = 0.0
    vel_value.angular.z = 0.0
    publisher.publish(vel_value)
    for i in range(len(down_path)):
        down_path[i][0],down_path[i][1] = down_path[i][0] /1000,(2000-down_path[i][1])/1000
    angles = []
    for i in range(len(down_path)-1):
        angle = calculate_theta_gazebo(down_path[i],down_path[i+1])
        angles.append(angle)
    distances = []
    for i in range(len(down_path)-1):
        distance = calculate_distance_gazebo(down_path[i],down_path[i+1])
        distances.append(distance)
    print("Angles for Gazebo",angles)
    print("Distances for Gazebo",distances)
    for i in range(len(down_path)):
        go_to_goal(angles[i],distances[i],down_path[i],vel_value,publisher,r)

def main():
    canvas = np.ones((2000,6000,3),dtype='uint8')
    canvas = obstacle_space(6000,2000,canvas,105,50) 
    image =  canvas
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,binary_image = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    start_x,start_y,goal_x,goal_y = get_start_goal_inputs(binary_image)
    #Variables
    start_y = 2000-start_y
    goal_y = 2000-goal_y
    start = (start_x,start_y)
    goal = (goal_x,goal_y)
  
    print("The Start Node selected is :",start)
    print("The Goal Node Selected is :", goal)
    distance_thresh = 40
    step_size = 80
    maxFailedAttempts = 300
    start = time.time()
    path,final_line, tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list = main_comb(binary_image, start_x,start_y, goal_x,goal_y, step_size, distance_thresh, maxFailedAttempts)
    end = time.time()
    
    print("The BTO-RRT final path :",path)
    print("The Time Taken for BTO RRT exploration:", end-start ,"seconds")
    
    down_path = down_path_sampling(binary_image, path ,goal_x,goal_y)
    print("Optimized Path after DownSample:",down_path)
    image = visualise(image,final_line,path,tree1_p_list,tree1_c_list,tree2_p_list,tree2_c_list,down_path)
    ros_visualisation(down_path)

            
    
    
if __name__ == '__main__':
    pose_x = 0.0
    pose_y = 0.0
    angle_z = 0.0
    main()
