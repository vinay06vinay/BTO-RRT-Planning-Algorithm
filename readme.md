# Project Members:
1. Anuj Zore - zoreanuj - 119198957
2. Vinay Krishna Bukka - vinay06 - 118176680

# File List In Zip Folder 

1. "vinay06_anuj_final_report.pdf" - project report
2. "vinay06_anuj_final.pptx" - final presentation
3. "vinay06_anuj_2D_video.avi","vinay06_anuj_gazebo_simulation.mp4" - 2D and 3D exploration video
4. "vinay06_anuj_final2Dcode.py"- 2D python code
5. "bto_gazebo" - Folder to run for ROS Noetic.
6. "map{i}.bmp" - different maps required for testing.

## Part - 1
### Instructions to Run the Code : 

1. From the extracted zip file name "vinay06_anuj_final_project.zip", take out the "vinay06_anuj_final2Dcode.py" file from folder and open any Python IDE or terminal
2. If Python IDE is used, please click on the run button
Note : Before running please make sure whether these libraries are installed in the system : numpy,cv2,matplotlib,time.
Any Library not present can be by typing below command in terminal window using pip:
       pip install numpy
       
### Input Format for Two Test Cases : Click on enter after entering a value at each prompt

#### Test Case 1 : 
	Enter the x-coordinate of start node
	 10
	Enter the y-coordinate of start node
	 10
	Enter the x-coordinate of goal node
	 490
	Enter the y-coordinate of goal node
	 490

4. Once the code is run, a few information about the code is printed on the console and the cv2 visualisation video is recorded and saved
5. Video will be saved under the name "BTO-RRT.avi". Alternatively, you can find the video in below links
6. Please make sure you have the maps given in zip folder in the same directory where the code is being run.

## Part - 2
### Instructions to Run the Code : 

#### Test Case
	Enter the x-coordinate of start node
	 10
	Enter the y-coordinate of start node
	 10
	Enter the x-coordinate of goal node
	 490
	Enter the y-coordinate of goal node
	 490
1. First, download the package from below path after extracting the zip
	* bto_gazebo
2. Keep this folder "bto_gazebo" inside the source folder of your respective ros workspace.
3. Perfrom below set of commands now on the terminal window
	* cd ~/catkin_ws
	* catkin_make
	* source devel/setup.bash
	* roslaunch bto_gazebo bto_gazebo.launch
	* rosrun bto_gazebo bto_gazebo.py (In another terminal)
4. A new terminal window will be opened where you need to enter the coordinates as in Test Case.
5. The turtlebot keeps moving to its goal point. The video link for the same is in the presentation slides and also in the zip folder.
	
