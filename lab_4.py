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

class Motor_Schema:
    global killer
    def __init__(self, sb_motor_publisher):
        self.sb_motor_publisher = sb_motor_publisher
        self.motor_pos = [0,0,0,0]
        self.global_goal = [0,0,0,0]
        self.global_speed = 10
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
                    
                    self.sb_motor_publisher.publish(twist)
                    time.sleep(0.1/speed)  
                    


class DancePerformance:
    global killer, dance_clock
    def __init__(self):
        # self.motors = motors
        self.danceSteps = []
        self.run = False
        self.freeStyle = FreeStyle(None)
        # self.sb_motors = 0
        x = threading.Thread(target=self.execute, args=())
        x.start()
        
    def start_dance(self):
        global dance_clock
        dance_clock = 0
        self.run = True
    def addDanceStep(self, danceStep, time):
        self.danceSteps.append([danceStep, time])
        # self.sb_motors = danceStep.motors
    def addFreeStyle(self, freeStyleDanceStep):
        self.freeStyle = freeStyleDanceStep
        # self.sb_motors = freeStyleDanceStep.motors
    
    def execute(self):
        global dance_clock
        while not self.run and not killer.kill_now:
            for actionStep in self.danceSteps:
                dance = actionStep[0]
                dance_release_time = actionStep[1]
                print("----------------------------------- Next step at: ",dance_release_time ," current time: ",dance_clock)
                while dance_clock < dance_release_time and not killer.kill_now:
                    # random_buddy = random.randint(0,3)
                    self.freeStyle.perform()
                    time.sleep(0.01)
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
    def motion_in_progress(self):
        for motor in self.motors:
            if(motor.inProgress()):
                return True
        return False    
        
    def perform(self):
        count = 0
        for step in self.steps:
            print("..... performing step #",count, step );
            goal = step[0:4]
            speed = step[4]
            for motor in self.motors: 
                motor.move(goal, speed)
            count = count + 1
            while(self.motion_in_progress() and not killer.kill_now):
                time.sleep(0.1)
                pass

class FreeStyle(DanceStep):
    def perform(self):
        steps = [[10, 0, 0, 20, 10], [5, 20, 0, 30, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5]]
        self.addMoves(steps)
        super().perform()
    # def perform(self, id):
        # if id>3:
        #     id = 0
        # steps = [[10, 0, 0, 20, 10], [5, 20, 0, 30, 5], [-10, 0, 0, 0, 15], [-5, 0, 0, 20, 5]]
        # for step in steps:
        #     goal = step[0:4]
        #     speed = step[4]
        #     self.motors[id].move(goal,speed)
        #     while(self.motion_in_progress() and not killer.kill_now):
        #         time.sleep(0.1)
        #         pass
        # for repeat in range(5):
        #     self.confused_random(id)
        

    def confused_random(self, id):
            rand = random.randint(0,40)
            goal = [0,0,0,0]
            goal[0] =  (-1)**rand*rand
            goal[1] =  (-1)**rand*rand
            goal[2] =  (-1)**rand*rand
            goal[3] =  (-1)**rand*rand
            
            print("random movement", goal)
            speed = random.randint(3,10)
            self.motors[id].move(goal, speed)

            goal = self.reverse_goal(goal)
            self.motors[id].move(goal, speed)

    def reverse_goal(self, goal):
        new_goal = [0,0,0,0]
        new_goal[0] = -goal[0]
        new_goal[0] = -goal[1]
        new_goal[0] = -goal[2]
        new_goal[0] = -goal[3]

        

class GenericBehavior(object):
    def __init__(self):
        
        self.sb_motors = []
        self.dance = DancePerformance()
        self.pub = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory, queue_size=20
        )
        self.sb_motor_publisher_0 = rospy.Publisher('/sb_0_cmd_state', TwistStamped, queue_size=1)
        self.sb_motor_publisher_1 = rospy.Publisher('/sb_1_cmd_state', TwistStamped, queue_size=1)
        self.sb_motor_publisher_2 = rospy.Publisher('/sb_2_cmd_state', TwistStamped, queue_size=1)
        self.sb_motor_publisher_3 = rospy.Publisher('/sb_3_cmd_state', TwistStamped, queue_size=1)


        self.sb_motors.append(Motor_Schema(self.sb_motor_publisher_0));
        self.sb_motors.append(Motor_Schema(self.sb_motor_publisher_1));
        self.sb_motors.append(Motor_Schema(self.sb_motor_publisher_2));
        self.sb_motors.append(Motor_Schema(self.sb_motor_publisher_3));
        
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
            danceStep = DanceStep(self.sb_motors)
            for eachMove in step_set:
                danceStep.addMoves(eachMove)
            simpleDanceSteps.append(danceStep)   
        count = 5
        
        self.dance.addFreeStyle(FreeStyle(self.sb_motors))

        for simpleStep in simpleDanceSteps:
            self.dance.addDanceStep(simpleStep, count)
            count = count * 2
        # self.dance.execute() 
        
        
        self.dance.start_dance()
        

    def callback_1(self, data):
        # print ("Callback 1: ",threading.current_thread())
        return
               


    def callback_2(self, data):
        # print ("callback 2: ",threading.current_thread())
        
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
