# rosbag_to_kitti

I temporarily added some codes to [kitti_to_rosbag](https://github.com/ethz-asl/kitti_to_rosbag) to convert/export a rosbag containing point cloud, image, imu, and GPS information to KITTI raw dataset format.
To run, first adjust the hardcoded GPS/IMU/camera transforms to velodyne and their corresponding topic names in kitti_live_node.cpp and then build and run it. Then play the rosbag file to get the exported raw data.
