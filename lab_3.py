#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage, Image
from moveit_msgs.msg import DisplayTrajectory
from threading import Thread
from geometry_msgs.msg import TwistStamped


# Python 2/3 compatibility imports
import sys
import copy
import random
import math
import rospy
import time
import geometry_msgs.msg
import moveit_commander
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import threading

motor_pos = [0, 0, 0, 0]
global_goal = [0,0,0,0]
global_speed = 1

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []
import signal
import time

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    self.kill_now = True

killer = GracefulKiller()

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String

twist = TwistStamped()
pause=True
my_time=time.time()
is_confused=0

dance_clock = time.time()


class Dance:
    global killer, dance_clock
    def __init__(self):
        # self.motors = motors
        self.danceSteps = []
        self.run = False
        
        x = threading.Thread(target=self.execute, args=())
        x.start()
        
    def start_dance(self):
        global dance_clock
        dance_clock = 0
        self.run = True
    def addDanceStep(self, danceStep, time):
        self.danceSteps.append([danceStep, time])
    
    def execute(self):
        global dance_clock
        while not self.run and not killer.kill_now:
            for actionStep in self.danceSteps:
                dance = actionStep[0]
                dance_release_time = actionStep[1]
                print("----------------------------------- Next step at: ",dance_release_time ," current time: ",dance_clock)
                while dance_clock < dance_release_time and not killer.kill_now:
                    time.sleep(0.1)
                    pass
                dance.perform()
        print("============================================ Thank you For Watching !!! ================================")
        print("                                                  Hope you Enjoyed                 ")
        
            

class DanceStep:
    global killer, dance_clock
    def __init__(self, motors):
        self.motors = motors
        self.steps = []
    def addMoves(self, step):
        self.steps.append(step)
    def perform(self):
        count = 0
        for step in self.steps:
            print("..... performing step #",count, step );
            goal = step[0:4]
            speed = step[4]
            self.motors.move(goal, speed)
            count = count + 1
            while(self.motors.inProgress() and not killer.kill_now):
                time.sleep(0.1)
                pass
        

class Motor_schema:
    global killer
    def __init__(self, buddy):
        self.buddy = buddy
        self.motor_pos = [0,0,0,0]
        self.global_goal = [0,0,0,0]
        self.global_speed = 0
        x = threading.Thread(target=self.motor_control, args=())
        x.start()
        
        
    
    def move(self, goal, speed):
        self.global_goal = goal
        self.global_speed = speed
        # self.motor_control()
    def inProgress(self):
        return self.motor_pos != self.global_goal

    def motor_control(self):
            print ("motor control thread: ",threading.current_thread())
            while not killer.kill_now:
                
                transient_pos = self.motor_pos
                goal = self.global_goal
                speed = self.global_speed
                if(goal != self.motor_pos): 
                    if(goal[0]>self.motor_pos[0]):
                        transient_pos[0] = transient_pos[0]+1
                    elif(goal[0]<self.motor_pos[0]):
                        transient_pos[0] = transient_pos[0]-1
                    
                    if(goal[1]>self.motor_pos[1]):
                        transient_pos[1] = transient_pos[1]+1
                    elif(goal[1]<self.motor_pos[1]):
                        transient_pos[1] = transient_pos[1]-1

                    if(goal[2]>self.motor_pos[2]):
                        transient_pos[2] = transient_pos[2]+1
                    elif(goal[2]<self.motor_pos[2]):
                        transient_pos[2] = transient_pos[2]-1

                    if(goal[3]>self.motor_pos[3]):
                        transient_pos[3] = transient_pos[3]+1
                    elif(goal[3]<self.motor_pos[3]):
                        transient_pos[3] = transient_pos[3]-1
                    print("Movement values: ",transient_pos)
                    self.motor_pos = transient_pos
                    
                    twist.twist.linear.x = self.motor_pos[0]
                    twist.twist.linear.y = self.motor_pos[1]
                    twist.twist.linear.z = self.motor_pos[2]
                    twist.twist.angular.x = self.motor_pos[3]
                    
                    self.buddy.servopub.publish(twist)
                    time.sleep(0.1/speed)  
                    


class GenericBehavior(object):
    def __init__(self):
        
        self.motors = Motor_schema(self)
        self.dance = Dance()
        self.pub = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory, queue_size=20
        )
        self.servopub = rospy.Publisher('/sb_cmd_state', TwistStamped, queue_size=1)
        self.talk = rospy.Publisher('/talker', String, queue_size=1)
        
        
        rospy.loginfo("Node started.")
        self.group_name = "survivor_buddy_head"
        # x = threading.Thread(target=self.motors.motor_control(), args=())
        # x.start()
        self.dancePerformance()
        
    def dancePerformance(self):
        simpleDanceSteps = [] 
        steps1 = [[0, 0, 0, 0, 2], [10, 0, 0, 20, 10], [5, 20, 0, 30, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5], [-5, -20, 0, 30, 10], [5, 0, 0, 0, 15], [0, 0, 10, 45, 10], [5, -10, -10, 0, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5], [-5, -20, 0, 30, 10],  [0, 0, -10, -45, 10], [5, 10, -10, 0, 5], [-5, -5, 10, 0, 10], [10, 0, 0, 20, 10], [5, 20, 0, 30, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5], [-5, -20, 0, 30, 10], [5, 0, 0, 0, 15], [0, 0, 10, 45, 10], [5, -10, -10, 0, 5], [-5, 5, 10, 0, 10]]
        steps2 = [[10, 0, 0, 20, 10], [5, 20, 0, 30, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5], [-5, -20, 0, 30, 10], [5, 0, 0, 0, 15], [0, 0, 10, 45, 10], [-5, 10, -10, 0, 5], [5, -5, 10, 0, 10]]
        steps3 = [[-5, 5, 10, 0, 10],  [10, 0, 0, 20, 10], [5, 20, 0, 30, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5], [-5, -20, 0, 30, 10], [5, 0, 0, 0, 15], [0, 0, 10, 45, 10], [5, -10, -10, 0, 5], [-5, 5, 10, 0, 10], [10, 0, 0, 20, 10], [5, 20, 0, 30, 5]]
        steps4 = [[5, 0, 0, 0, 15], [0, 0, 10, 45, 10], [-5, 10, -10, 0, 5], [5, -5, 10, 0, 10], [-10, 0, 0, 0, 15]]

        steps = []
        steps.append(steps1)
        steps.append(steps2)
        steps.append(steps3)
        steps.append(steps4)

        for step_set in steps:
            danceStep = DanceStep(self.motors)
            for eachMove in step_set:
                danceStep.addMoves(eachMove)
            simpleDanceSteps.append(danceStep)   
        count = 5
        for simpleStep in simpleDanceSteps:
            self.dance.addDanceStep(simpleStep, count)
            count = count * 2
        # self.dance.execute() 
        self.dance.start_dance()
        
        
    def confused_random(self):
                rand = random.randint(0,10)
                print("Confused")
                #joint_goal = self.move_group.get_current_joint_values()
                if rand == 0:
                    #joint_goal[0] =  (-1)**rand * pi/10 # pi/6 # Enter a value
                    twist.twist.linear.x =  (-1)**rand
                elif rand == 1:
                    #joint_goal[1] =  (-1)**rand * pi/10 # pi/6 # Enter a value
                    twist.twist.linear.y =  (-1)**rand
                elif rand == 2:
                    #joint_goal[2] =  (-1)**rand * pi/10 # pi/6 # Enter a value
                    twist.twist.linear.z =  (-1)**rand
                elif rand == 3:
                    #joint_goal[3] =  (-1)**rand * pi/10 # pi/6 # Enter a value
                    twist.twist.angular.x =  (-1)**rand
                if rand < 4:
                    print("random movement")
                    self.servopub.publish(twist)
                    #self.move_group.go(joint_goal, wait=True)
                    #plan = self.move_group.plan()
                    #self.move_group.stop()
                
                    #original_goal = self.move_group.get_current_joint_values()
                    #original_goal[0] =  0 # Enter a value
                    #original_goal[1] =  0 # Enter a value
                    #original_goal[2] =  0 # Enter a value
                    #original_goal[3] =  0 # Enter a value
                
                    #self.move_group.go(original_goal, wait=True)
                    #plan = self.move_group.plan()
                    #self.move_group.stop()    
                    #self.move_group.execute(plan[1], wait=True)
                else:
                    twist.twist.linear.x = 0
                    twist.twist.linear.y = 0
                    twist.twist.linear.z = 0
                    twist.twist.angular.x = 0

    def pos_equal(self, motor_pos, goal):
        if(motor_pos[0] == goal[0] and motor_pos[1] == goal[1] and motor_pos[2] == goal[2] and motor_pos[3] == goal[3]):
            return True
        else:
            return False

    def callback_1(self, data):
        # print ("Callback 1: ",threading.current_thread())
        return
               


    def callback_2(self, data):
        # print ("callback 2: ",threading.current_thread())
        global pause, motor_pos, global_goal, global_speed
        global my_time
        global is_confused
        global recognizer, options, base_options, results, images

        if time.time()<=0.4+my_time:
            return
        my_time=time.time()

        np_arr = np.frombuffer(data.data,np.uint8)
        image = cv2.imdecode(np_arr, 1)
        cv2.namedWindow("Image Window",1)
        
        
        
        # STEP 3: Load the input image.
        #mpimage = mp.Image.create_from_file(data.data)
        mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        cv2.imshow("Image Window",image)
        cv2.waitKey(3)

    # STEP 4: Recognize gestures in the input image.
        recognition_result = recognizer.recognize(mpimage)

        # STEP 5: Process the result. In this case, visualize it.
        #images.append(image)
        top_gesture = recognition_result.gestures
        hand_landmarks = recognition_result.hand_landmarks
        #print("====",top_gesture)
        with mp_hands.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=0.6) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))           
            if results.multi_hand_landmarks:
                x =[]
                y =[]
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx,coord in enumerate(hand_landmarks.landmark):
                        x.append(coord.x)
                        y.append(coord.y)
                #Reffered from Sambhav
                finger = (x[4]-x[8])*(x[4]-x[8]) + (y[4]-y[8])*(y[4]-y[8])   +   (x[12]-x[8])*(x[12]-x[8]) + (y[12]-y[8])*(y[12]-y[8])   +   (x[12]-x[16])*(x[12]-x[16]) + (y[12]-y[16])*(y[12]-y[16]) +   (x[20]-x[16])*(x[20]-x[16]) + (y[20]-y[16])*(y[20]-y[16])
                finger=finger*1000
                
                
                
                finger2= (x[12]-x[4])*(x[12]-x[4]) + (y[12]-y[4])*(y[12]-y[4])   +   (x[12]-x[16])*(x[12]-x[16]) + (y[12]-y[16])*(y[12]-y[16]) +  (x[20]-x[16])*(x[20]-x[16]) + (y[20]-y[16])*(y[20]-y[16])  
                finger2=finger2*1000
                thumb =    (x[4]-x[8])*(x[4]-x[8]) + (y[4]-y[8])*(y[4]-y[8]) 
                thumb=thumb*100 
                print(finger, finger2,  max(x)*100-min(x)*100, thumb)
                
                if finger<8 and thumb<10:
                    self.talk.publish("STOP")
                    print("Video will be  stopped")
                    goal = [0,0,0,0]
                    speed = 1
                    self.motors.move(goal, speed)
                    is_confused=0
                elif max(x)*100-min(x)*100<9 and not pause:
                    print("Video will be paused")
                    self.talk.publish("PAUSE")
                    pause=not  pause
                    is_confused=0
                    my_time=my_time+0.3
                    goal = [-10, -10, -10, -10]
                    speed = 2
                    self.motors.move(goal, speed)
                elif max(x)*100-min(x)*100<9 and pause:
                    print("Video will be played")
                    self.talk.publish("PLAY")
                    pause=not pause
                    is_confused=0
                    my_time=my_time+0.3
                elif finger2<17 and x[8]<0.4:
                    print("Raise video's Volumne")
                    self.talk.publish("RAISE")
                elif finger2<17 and x[8]>0.6:
                    print("Decrease the volume now!")
                    self.talk.publish("LOWER")
                    is_confused=0
                elif finger2<8:
                    print("DECIDING on volume")
                elif max(x)<0.33:
                    self.talk.publish("FORWARD")
                    print("10 sec forward")
                elif min(x)>0.67:
                    print("10 sec backward")
                    self.talk.publish("BACKWARD")
                else:
                    is_confused=is_confused+1
                    print("Gesture not recognized!!!!!!!!!!!!!!!!!!!")
            if is_confused>0:
                # self.confused_random()
                goal = [20, 20, 20, 20]
                speed = 10
                self.motors.move(goal, speed)
                # motor_control(self, global_goal, 1)
                
                print("I'm confused!")
                is_confused=0
            else:
                pass
                # motor_pos = [0,0,0,0]        
        pass

def clock_update(event):
    global dance_clock
    dance_clock = dance_clock + 0.1
    print("                  clock updated", dance_clock)

if __name__ == '__main__':
    
    rospy.init_node("lab_1_node")
    moveit_commander.roscpp_initialize(sys.argv)
    buddy = GenericBehavior()
    camera_sub = rospy.Subscriber("/camera/image/compressed", CompressedImage, callback=buddy.callback_2, queue_size=1)
    audio_sub = rospy.Subscriber("/audio", Float32MultiArray, callback= buddy.callback_1, queue_size=1) 
    # print ("main ",threading.current_thread())
    # init_pos = [0,0,0,0]
    # x = threading.Thread(target=self.motor_control, args=(global_goal, 1))
    # x.start()

    # motor_control(buddy, global_goal, 1);
    # killer = GracefulKiller()
    # motors = Motor_schema(buddy)
    # x = threading.Thread(target=motors.move, args=())
    # x.start()
    
    rospy.Timer(rospy.Duration(0.1), clock_update)

    rospy.spin()
    # rospy.MultiThreadedSpinner spinner(4); # Use 4 threads
    # spinner.spin(); # spin() will not return until the node has been shutdown
