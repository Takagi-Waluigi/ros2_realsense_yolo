from setuptools import find_packages, setup

package_name = 'ros2_realsense_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_publisher = ros2_realsense_yolo.pose_publisher:main',
            'laser_publisher = ros2_realsense_yolo.laser_publisher:main',
            'user_detection_publisher = ros2_realsense_yolo.user_detection_publisher:main',
        ],
    },
)
