import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3

KEYPOINTS_NAMES = [
    "nose",  # 0
    "eye(L)",  # 1
    "eye(R)",  # 2
    "ear(L)",  # 3
    "ear(R)",  # 4
    "shoulder(L)",  # 5
    "shoulder(R)",  # 6
    "elbow(L)",  # 7
    "elbow(R)",  # 8
    "wrist(L)",  # 9
    "wrist(R)",  # 10
    "hip(L)",  # 11
    "hip(R)",  # 12
    "knee(L)",  # 13
    "knee(R)",  # 14
    "ankle(L)",  # 15
    "ankle(R)",  # 16
]


class PosePublisherClass(Node):

    def __init__(self):
        self.model = YOLO("yolov8n-pose.pt")

        super().__init__('simple_face_dectation')

        self.enableLog = False

        # デプスとカラー画像の設定
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()

        ctx = rs.context() # Create librealsense context for managing devices
        serials = []
        if (len(ctx.devices) > 0):
            for dev in ctx.devices:
                print ('Found device: ', \
                        dev.get_info(rs.camera_info.name), ' ', \
                        dev.get_info(rs.camera_info.serial_number))
                serials.append(dev.get_info(rs.camera_info.serial_number))
        else:
            print("No Intel Device connected")

        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                self.arrayPublisher_ = self.create_publisher(PoseArray, 'body', 10)
                
                break
        if not self.found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        #シリアル番号を指定する
        self.selialNumber = '048522074360'

        self.config.enable_device(self.selialNumber)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # ストリーミング開始
        self.pipeline.start(self.config)

        # ストリーミング情報の取得
        self.profile = self.pipeline.get_active_profile()
        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        # ポイントクラウドの設定
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)
        self.colorizer = rs.colorizer()

        # ループ処理
        timer_period = 0.03
        self.timer = self.create_timer(timer_period, self.loop)    

    def loop(self):
        self.msg = Vector3()
        
        # フレーム入力待ち
        self.frames = self.pipeline.wait_for_frames()

        #フレーム入力時の処理
        self.aligned_frames = self.align.process(self.frames)
        self.depth_frame = self.aligned_frames.get_depth_frame()
        self.color_frame = self.aligned_frames.get_color_frame()

        self.depth_frame = self.decimate.process(self.depth_frame)

        self.depth_intrinsics = rs.video_stream_profile(
                        self.depth_frame.profile).get_intrinsics()
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        
        #imageをnumpy arrayに
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = np.asanyarray(self.color_frame.get_data())

        # ポイントクラウド関連処理
        self.points = self.pc.calculate(self.depth_frame)
        self.v= self.points.get_vertices()
        self.verts = np.asanyarray(self.v).view(np.float32).reshape(-1, 3)  # xyz
        
        # リサイズ
        self.color_image_s = cv2.resize(self.color_image, (self.w, self.h))  

        #YOLO処理
        self.results = self.model(self.color_image_s, show=False, save=False)
        self.names = self.results[0].names
        self.classes = self.results[0].boxes.cls
        self.boxes = self.results[0].boxes
        self.annotatedFrame = self.results[0].plot()
        self.keypoints = self.results[0].keypoints  
        
        self.id = 1
        self.layoutArray = []
        self.positionArray=[]
        
        # 検出された人数分のキーポイントの処理を行うfor文
        for kp in self.keypoints:
            self.msg_array = Float32MultiArray()
            
            if(kp.conf is not None):
                if self.enableLog:
                    print("ID:",self.id)
                
                kpxy=kp.xyn.tolist()
                kpconf=kp.conf.tolist()            
            
                # キーポイントの座標の取得
                for index in range(len(kpconf[0])):
                    self.dimInfo = MultiArrayDimension()
                    self.dimInfo.label = str(self.id) + "_" + KEYPOINTS_NAMES[index] 
                    self.dimInfo.stride = 4
                    self.layoutArray.append(self.dimInfo)
                    
                    self.target_x = int(kpxy[0][index][0] * self.w)
                    self.target_y = int(kpxy[0][index][1] * self.h)  

                    #ポイントクラウド上の座標の取得      
                    self.target_v = self.verts[self.target_x + (self.target_y - 1)* self.w]

                    self.poses = Pose()
                    self.poses.position.x = float(self.target_v[0])
                    self.poses.position.y = float(self.target_v[1])
                    self.poses.position.z = float(self.target_v[2])

                    self.poses.orientation.x = float(kpconf[0][index])

                    self.positionArray.append(self.poses)

                    if self.enableLog:
                        print(self.id, ":world position:" + KEYPOINTS_NAMES[index] + ":", self.target_v)

                self.headerInfo = Header()
                self.headerInfo.frame_id = str(self.id)

                self.msg_array = PoseArray(header=self.headerInfo, poses=self.positionArray)            
                self.arrayPublisher_.publish(self.msg_array)

                self.id += 1
            
            else:
                print("[DEBUG] Keypoints is NOT detecting.")


def main(args=None):
    rclpy.init(args=args)

    publisher = PosePublisherClass()

    rclpy.spin(publisher)
    
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
