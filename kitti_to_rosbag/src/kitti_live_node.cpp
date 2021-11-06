/*
Copyright (c) 2016, Helen Oleynikova, ETH Zurich, Switzerland
You can contact the author at <helen dot oleynikova at mavt dot ethz dot ch>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of ETHZ-ASL nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ETHZ-ASL BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <tf_conversions/tf_eigen.h>
#include "tf/transform_datatypes.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf/transform_broadcaster.h>


#include <rosgraph_msgs/Clock.h>
#include "kitti_to_rosbag/kitti_parser.h"
#include "kitti_to_rosbag/kitti_ros_conversions.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf/transform_listener.h>


#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <ctime>
#include <sstream>
#include <chrono>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <experimental/filesystem>
#include <signal.h>
#include <unistd.h>

using namespace std::chrono;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

namespace fs = boost::filesystem;

namespace kitti {

class KittiLiveNode {
 public:
  KittiLiveNode(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
                const std::string& calibration_path,
                const std::string& dataset_path);

  // Creates a timer to automatically publish entries in 'realtime' versus
  // the original data,
  typedef message_filters::sync_policies::ApproximateTime<Image, 
                                                          PointCloud2,
                                                          NavSatFix,
                                                          Imu,
                                                          geometry_msgs::TwistWithCovarianceStamped> ApproxSyncPolicy;
  void startPublishing(double rate_hz);

  bool publishEntry(uint64_t entry);
  
  bool sync_callback(const ImageConstPtr& image, 
                     const PointCloud2ConstPtr& pointcloud,
                     const NavSatFixConstPtr& fix,
                     const ImuConstPtr& imu,
                     const geometry_msgs::TwistWithCovarianceStampedConstPtr& twist);

  void publishTf(uint64_t timestamp_ns, const Transformation& imu_pose);
    
  void publishClock(uint64_t timestamp_ns);  
  
  string fmtTransToKittiCalib(tf::Transform transform);

 private:
  void timerCallback(const ros::WallTimerEvent& event);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  
  tf::TransformListener* listener_;
  tf::Transform velo2CamTrnasform_;
  tf::Transform velo2ImuTrnasform_;
  tf::Transform velo2GpsTransform_;

  Eigen::IOFormat* commaInitFmt;  
  string formatMatrix3x3(const tf::Matrix3x3& matrix3x3);
  string formatVector3(const tf::Vector3& vector3);
  string formatTimestamp(uint32_t sec, uint32_t nsec);
  tf::Matrix3x3 toTFMat(const cv::Mat mat);

  bool checkAndCreateFile(std::string path, std::ofstream& file);
  void toTransform(double x, double y, double z, double roll, double pitch, double yaw, tf::Transform& transform);
  void writeRawCalibFiles(float* S_02, tf::Matrix3x3 K_02, float* D_02, float* S_rect_02, tf::Transform P_rect_02);
  
  void writeDetectionCalibs(const std::string& detectionCalibPath, const tf::Transform& P2,
                                         const tf::Transform& P0, const tf::Transform& P1, const tf::Transform& P3);
  void writeOdometryCalib(const std::string& odometryCalibPath, const tf::Transform& P2,
                                         const tf::Transform& P0, const tf::Transform& P1, const tf::Transform& P3);
  void undistortImage(const cv::Mat& image, cv::Mat& image_undistort, cv::Mat cameraMatrix, cv::Mat distorsionCoeff,
                      cv::Size imageSize, cv::Mat& newCameraMatrix, cv::Size& newImageSize);
  
  string calib_time_;
  cv::Mat cameraMatrix;
  cv::Mat distorsionCoeff;
  cv::Size imageSize;
  // TODO(farzad) read calibration params from CameraInfo topic.
  float* S_02;
  float* D_02;
  tf::Vector3 cam02_K_origin;
  tf::Matrix3x3 cam02_K_basis;
  bool isRectified;
  bool isCalibDone;
  
  // Followings are initialized after rectification in sync_callback
  cv::Mat cameraMatrixRectified;
  cv::Size imageSizeRectified;
  
  // From iter3 calibration results:
  const double cam2Velo_roll_ = -1.5104;
  const double cam2Velo_pitch_ = 0.1128;
  const double cam2Velo_yaw_ = -1.5257;
  const double cam2Velo_x_ = 0.0426;
  const double cam2Velo_y_ = 0.2125;
  const double cam2Velo_z_ = 0.0578;
  
  // Home folders for different tasks with pre-processed data
  const std::string odometryFolder = "odometry";
  const std::string detectionFolder = "detection";
  const std::string trackingFolder = "tracking";
  const std::string odometrySeqID = "00";

  // Publishers for the topics.
  ros::Publisher clock_pub_;
  ros::Publisher pointcloud_pub_;
  ros::Publisher pose_pub_;
  ros::Publisher transform_pub_;
  tf::TransformBroadcaster tf_broadcaster_;

  image_transport::ImageTransport image_transport_;
  std::vector<image_transport::CameraPublisher> image_pubs_;

  // Decides the rate of publishing (TF is published at this rate).
  // Call startPublishing() to turn this on.
  ros::WallTimer publish_timer_;

  kitti::KittiParser parser_;

  std::string world_frame_id_;
  std::string imu_frame_id_;
  std::string cam_frame_id_prefix_;
  std::string velodyne_frame_id_;

  uint64_t current_entry_;
  uint64_t publish_dt_ns_;
  uint64_t current_timestamp_ns_;
  uint64_t base_timestamp_ns_;
  
  std::string dataset_path_;
  std::string cameraDataFolder;
  std::string lidarDataFolder;
  std::string gpsDataFolder; 
  std::string poseTumDataFolder;
  std::string imageTimestampPath;
  std::string pcTimestampPath;
  std::string poseTimestampPath;

  std::string odometryPoseFolder;
  std::string odometrySeqXFolder;
  std::string odometryVeloFolder;
  std::string odometryPosesPath;
  std::string odometryTimesPath;

  std::string detectionCalibFolder;
  std::string detectionImageFolder;
  std::string detectionVelodyneFolder;

  std::ofstream imageTimestampFile;
  std::ofstream pcTimestampFile;
  std::ofstream poseTimestampFile;
  std::ofstream poseKittiFile;
  
  std::ofstream odometryPosesFile;
  std::ofstream odometryTimesFile;
  

  std::ofstream detectionCalibFile;
  
  std::unique_ptr<message_filters::Subscriber<Image>> img_sub;
  std::unique_ptr<message_filters::Subscriber<PointCloud2>> cloud_sub;
  std::unique_ptr<message_filters::Subscriber<NavSatFix>> fix_sub;
  std::unique_ptr<message_filters::Subscriber<Imu>> imu_sub;
  std::unique_ptr<message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>> vel_sub;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync;
  
};

KittiLiveNode::KittiLiveNode(const ros::NodeHandle& nh,
                             const ros::NodeHandle& nh_private,
                             const std::string& calibration_path,
                             const std::string& dataset_path)
    : nh_(nh),
      nh_private_(nh_private),
      image_transport_(nh_),
      parser_(calibration_path, dataset_path, true),
      world_frame_id_("world"),
      imu_frame_id_("imu"),
      cam_frame_id_prefix_("cam"),
      velodyne_frame_id_("velodyne"),
      current_entry_(0),
      publish_dt_ns_(0),
      current_timestamp_ns_(0),
      base_timestamp_ns_(0),
      dataset_path_(dataset_path),
      listener_(new tf::TransformListener){
        
  // Load all the timestamp maps and calibration parameters.
  // parser_.loadCalibration();
  // parser_.loadTimestampMaps();

  // Advertise all the publishing topics for ROS live streaming.
  // clock_pub_ = nh_.advertise<rosgraph_msgs::Clock>("/clock", 1, false);
  // pointcloud_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
  //     "velodyne_points", 10, false);
  // pose_pub_ =
  //     nh_.advertise<geometry_msgs::PoseStamped>("pose_imu", 10, false);
  // transform_pub_ = nh_.advertise<geometry_msgs::TransformStamped>(
  //     "transform_imu", 10, false);

  // for (size_t cam_id = 0; cam_id < parser_.getNumCameras(); ++cam_id) {
  //   image_pubs_.push_back(
  //       image_transport_.advertiseCamera(getCameraFrameId(cam_id), 1));
  // }

  commaInitFmt = new Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");
  
  // Raw data paths
  cameraDataFolder = dataset_path_ + "/raw/" + KittiParser::kCameraFolder + "02" + "/" + KittiParser::kDataFolder;
  lidarDataFolder = dataset_path_ + "/raw/" + KittiParser::kVelodyneFolder + "/" + KittiParser::kDataFolder;
  gpsDataFolder = dataset_path_ + "/raw/" + KittiParser::kPoseFolder + "/" + KittiParser::kDataFolder;
  poseTumDataFolder = dataset_path_ + "/raw/" + "pose_tum" + "/" + KittiParser::kDataFolder;
  imageTimestampPath = dataset_path_ + "/raw/" + KittiParser::kCameraFolder + "02" + "/" + KittiParser::kTimestampFilename;
  pcTimestampPath = dataset_path_ + "/raw/" + KittiParser::kVelodyneFolder + "/" + KittiParser::kTimestampFilename;
  poseTimestampPath = dataset_path_ + "/raw/" + KittiParser::kPoseFolder + "/" + KittiParser::kTimestampFilename;

  // Odometry data paths
  odometryPoseFolder = dataset_path_ + "/" + odometryFolder + "/" + "poses";
  odometrySeqXFolder = dataset_path_ + "/" + odometryFolder + "/" + "sequences/" + odometrySeqID;
  odometryVeloFolder = odometrySeqXFolder + "/" "velodyne";

  odometryPosesPath = odometryPoseFolder + "/" + odometrySeqID + ".txt";
  odometryTimesPath = odometrySeqXFolder + "/" + "times.txt";
  

  // Detection data paths
  detectionCalibFolder = dataset_path_ + "/" + detectionFolder + "/" + "calib";
  detectionImageFolder = dataset_path_ + "/" + detectionFolder + "/" + "image_2";
  detectionVelodyneFolder = dataset_path_ + "/" + detectionFolder + "/" + "velodyne";

  try {
    fs::create_directories(cameraDataFolder);
    fs::create_directories(lidarDataFolder);
    fs::create_directories(gpsDataFolder);
    fs::create_directories(odometryPoseFolder);
    fs::create_directories(odometrySeqXFolder);
    fs::create_directories(odometryVeloFolder);
    fs::create_directories(detectionCalibFolder);
    fs::create_directories(detectionImageFolder);
    fs::create_directories(detectionVelodyneFolder);
  }
  catch (std::exception& e) { // Not using fs::filesystem_error since std::bad_alloc can throw too.
    std::cout << e.what() << std::endl;
  }
  
  imageTimestampFile.open(imageTimestampPath, std::ios_base::app);
  pcTimestampFile.open(pcTimestampPath, std::ios_base::app);
  poseTimestampFile.open(poseTimestampPath, std::ios_base::app);
  
  odometryPosesFile.open(odometryPosesPath, std::ios_base::app);
  odometryTimesFile.open(odometryTimesPath, std::ios_base::app);


  ROS_INFO("Subscribing to topics...");
  img_sub.reset(new message_filters::Subscriber<Image>(nh_, "/camera/color/image_raw", 100));
  cloud_sub.reset(new message_filters::Subscriber<PointCloud2>(nh_, "/points_raw", 100));
  fix_sub.reset(new message_filters::Subscriber<NavSatFix>(nh_, "/fix", 100));
  imu_sub.reset(new message_filters::Subscriber<Imu>(nh_, "/imu/data", 100));
  vel_sub.reset(new message_filters::Subscriber<geometry_msgs::TwistWithCovarianceStamped>(nh_, "/ublox/fix_velocity", 100));
  sync.reset(new message_filters::Synchronizer<ApproxSyncPolicy>(ApproxSyncPolicy(100), *img_sub, *cloud_sub, *fix_sub, *imu_sub, *vel_sub));
  sync->registerCallback(boost::bind(&KittiLiveNode::sync_callback, this, _1, _2, _3, _4, _5));

  // Initialize camera intrinsics ----------
  calib_time_ = "08-Oct-2021 14:00:00";
  isRectified = false;
  isCalibDone = false;
  // TODO(farzad) read calibration params from CameraInfo topic.
  S_02 = new float[2]{640, 480};
  D_02 = new float[5]{0.167995, -0.549076, 0.000874, 0.000023, 0.493965};
  cam02_K_origin =  tf::Vector3(0, 0, 0);
  cam02_K_basis = tf::Matrix3x3(602.006653,    0.,         311.791565, \
                                0.,            601.831604, 249.589355, \
                                0.,            0.,         1.          );
  imageSize = cv::Size(S_02[0], S_02[1]);

  tf::Transform K_02 = tf::Transform(cam02_K_basis, cam02_K_origin);
  cameraMatrix = cv::Mat(3, 3, CV_64FC1);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      cameraMatrix.at<double>(i,j) = cam02_K_basis.getRow(i)[j];
  
  distorsionCoeff = cv::Mat(5, 1, CV_64FC1);
  for (int i = 0; i < 5; i++)
    distorsionCoeff.at<double>(i, 0) = D_02[i];
    
  // Initialize velo2Cam transform ----------
  tf::Transform cam2VeloTransform;
  toTransform(cam2Velo_x_, cam2Velo_y_, cam2Velo_z_, cam2Velo_roll_, cam2Velo_pitch_, cam2Velo_yaw_, cam2VeloTransform);
  velo2CamTrnasform_ = cam2VeloTransform.inverse();
  cout << "Tr_velo_to_cam: " << fmtTransToKittiCalib(velo2CamTrnasform_) << endl;

  // Initialize velo2IMU transform ----------
  tf::StampedTransform cam2ImuTransform;
  ROS_INFO("Waiting for camera to IMU transform...");
  while (nh_.ok()){
    try
    {
      listener_->lookupTransform("camera_color_optical_frame", "camera_imu_optical_frame", ros::Time(0), cam2ImuTransform);
      velo2ImuTrnasform_ = velo2CamTrnasform_ * cam2ImuTransform;
      cout << "Tr_velo_to_imu: " << fmtTransToKittiCalib(velo2ImuTrnasform_) << endl;
      ROS_INFO("Received!");  
      break;
    }
    catch(tf::TransformException &e)
    {
      ROS_WARN("%s",e.what());
      ros::Duration(1.0).sleep();
      continue;
    }
  }
  // Initialize gps2Velo transform ----------
  // From iter3 manual measurements:
  const double velo2Gps_roll = 0;
  const double velo2Gps_pitch = 0;
  const double velo2Gps_yaw = 0;
  const double velo2Gps_x = -0.39;
  const double velo2Gps_y = 0;
  const double velo2Gps_z = -0.1;
  
  toTransform(velo2Gps_x, velo2Gps_y, velo2Gps_z, velo2Gps_roll, velo2Gps_pitch, velo2Gps_yaw, velo2GpsTransform_);
  cout << "Tr_velo_to_gps: " << fmtTransToKittiCalib(velo2GpsTransform_) << endl; 
}

void KittiLiveNode::writeOdometryCalib(const std::string& odometryCalibPath, const tf::Transform& P2,
                                       const tf::Transform& P0 = tf::Transform::getIdentity(),
                                       const tf::Transform& P1 = tf::Transform::getIdentity(),
                                       const tf::Transform& P3 = tf::Transform::getIdentity()){
  std::ofstream odometryCalibFile;
  odometryCalibFile.open(odometryCalibPath, std::ios::out);
  odometryCalibFile << "P2: " << fmtTransToKittiCalib(P2);
  // odometryCalibFile << "P1: " << fmtTransToKittiCalib(P1);
  // odometryCalibFile << "P2: " << fmtTransToKittiCalib(P2);
  // odometryCalibFile << "P3: " << fmtTransToKittiCalib(P3);
  odometryCalibFile << "Tr: " << fmtTransToKittiCalib(velo2CamTrnasform_);
  odometryCalibFile.close();

}
void KittiLiveNode::writeDetectionCalibs(const std::string& detectionCalibPath, const tf::Transform& P2,
                                         const tf::Transform& P0 = tf::Transform::getIdentity(),
                                         const tf::Transform& P1 = tf::Transform::getIdentity(),
                                         const tf::Transform& P3 = tf::Transform::getIdentity()){
                            
  tf::Matrix3x3 R0_rect = tf::Matrix3x3::getIdentity();
  detectionCalibFile.open(detectionCalibPath, std::ios::out);

  detectionCalibFile << "P2: " << fmtTransToKittiCalib(P2);
  // detectionCalibFile << "P1: " << fmtTransToKittiCalib(P1);
  // detectionCalibFile << "P2: " << fmtTransToKittiCalib(P2);
  // detectionCalibFile << "P3: " << fmtTransToKittiCalib(P3);
  detectionCalibFile << "R0_rect: " << formatMatrix3x3(R0_rect);
  detectionCalibFile << "Tr_velo_to_cam: " << fmtTransToKittiCalib(velo2CamTrnasform_);
  detectionCalibFile << "Tr_imu_to_velo: " << fmtTransToKittiCalib(velo2ImuTrnasform_.inverse());
  detectionCalibFile << "Tr_gps_to_velo: " << fmtTransToKittiCalib(velo2GpsTransform_.inverse());
  detectionCalibFile.close();
}
void KittiLiveNode::toTransform(double x, double y, double z, double roll, double pitch, double yaw, tf::Transform& transform)
{
  transform.getOrigin().setValue(x, y,  z);
  transform.getBasis().setRPY(roll, pitch, yaw);
}

bool KittiLiveNode::sync_callback(const ImageConstPtr& image_msg,
                                  const PointCloud2ConstPtr& pc_msg,
                                  const NavSatFixConstPtr& fix_msg,
                                  const ImuConstPtr& imu_msg,
                                  const geometry_msgs::TwistWithCovarianceStampedConstPtr& twist_msg){

  std::cout << "======================== Current entry:" << current_entry_ << " ========================" << endl;

  std::string baseFileName = (boost::format("%06lu") % current_entry_).str();
  
  // Publish raw images.
  cv_bridge::CvImageConstPtr image_cv;
  image_cv = cv_bridge::toCvCopy(image_msg, image_encodings::BGR8);
  cv::Mat img_rgb = image_cv->image;
	std::string imageFrameEntry =   cameraDataFolder + "/" + baseFileName + ".png";
  imwrite(imageFrameEntry, img_rgb);          
  
  // Publish rectified images.
  cv::Mat img_rect;
  undistortImage(img_rgb, img_rect, cameraMatrix, distorsionCoeff, imageSize, cameraMatrixRectified, imageSizeRectified);
  cout << "------------- New_camera_matrix -------------" << endl;
  cout << cameraMatrixRectified << endl;
  cout << "newImageSize " << imageSizeRectified << endl;

  std::string imageRectFrameEntry =   detectionImageFolder + "/" + baseFileName + ".png";
  imwrite(imageRectFrameEntry, img_rect);          
  
  string imageTimestamp = formatTimestamp(image_msg->header.stamp.sec, image_msg->header.stamp.nsec);
  imageTimestampFile << imageTimestamp << std::endl;

  // Publish velodynes.
  std::string lidarFrameEntry =  lidarDataFolder + "/" + baseFileName + ".bin";
  
  std::ofstream pcKittiFile(lidarFrameEntry, std::ios::out | std::ios::binary | std::ios::app);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::fromROSMsg(*pc_msg, *pcl_cloud);
	
  for (size_t i = 0; i < pcl_cloud->points.size (); ++i)
  {
  	pcKittiFile.write((char*)&pcl_cloud->points[i].x, 3 * sizeof(float)); 
    pcKittiFile.write((char*)&pcl_cloud->points[i].intensity, sizeof(float));
  }
  pcKittiFile.close();
  
  string pcTimestamp = formatTimestamp(pc_msg->header.stamp.sec, pc_msg->header.stamp.nsec);
  pcTimestampFile << pcTimestamp << std::endl;
  
  // Publish GPS/IMU.
  std::string poseFrameEntry =  gpsDataFolder + "/" + baseFileName + ".txt";
  std::string poseTUMFrameEntry = poseTumDataFolder + "/" + baseFileName + ".txt";
  
  // lat:   latitude of the oxts-unit (deg)
  // lon:   longitude of the oxts-unit (deg)
  // alt:   altitude of the oxts-unit (m)
  double lat = fix_msg->latitude;
  double lon = fix_msg->longitude;
  double alt = fix_msg->altitude;
  
  // roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
  // pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
  // yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
  tf::Quaternion imu_quat;
  // tf2::convert(imu_msg->orientation , imu_quat);
  tf::quaternionMsgToTF(imu_msg->orientation, imu_quat);
  double roll, pitch, yaw;
  tf::Matrix3x3(imu_quat).getRPY(roll, pitch, yaw);
  
  // vn:    velocity towards north (m/s)
  // ve:    velocity towards east (m/s)
  // ve:    velocity towards east (m/s)
  // vf:    forward velocity, i.e. parallel to earth-surface (m/s)
  // vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
  double vn, ve, vf, vl, vu;
  vn = ve = 0;
  vf = twist_msg->twist.twist.linear.x;
  vl = twist_msg->twist.twist.linear.y;
  vu = twist_msg->twist.twist.linear.z;
   
  // ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
  // ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
  // az:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
  // af:    forward acceleration (m/s^2)
  // al:    leftward acceleration (m/s^2)
  // au:    upward acceleration (m/s^2)
  double ax, ay, az, af, al, au;
  ax = imu_msg->linear_acceleration.x;
  ay = imu_msg->linear_acceleration.y;
  az = imu_msg->linear_acceleration.z;
  af = al = au = 0;

  // wx:    angular rate around x (rad/s)
  // wy:    angular rate around y (rad/s)
  // wz:    angular rate around z (rad/s)
  // wf:    angular rate around forward axis (rad/s)
  // wl:    angular rate around leftward axis (rad/s)
  // wu:    angular rate around upward axis (rad/s)
  double wx, wy, wz, wf, wl, wu;
  wx = imu_msg->angular_velocity.x;
  wy = imu_msg->angular_velocity.y;
  wz = imu_msg->angular_velocity.z;
  wf = wl = wu = 0;
  
  // pos_accuracy:  velocity accuracy (north/east in m)
  // vel_accuracy:  velocity accuracy (north/east in m/s)
  // navstat:       navigation status (see navstat_to_string)
  // numsats:       number of satellites tracked by primary GPS receiver
  // posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
  // velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
  // orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)
  poseKittiFile.open(poseFrameEntry, std::ios::out);
  double pos_accuracy = 0, vel_accuracy = 0, navstat = 0, numsats = 0, posmode = 0, velmode = 0, orimode = 0;
  poseKittiFile << lat << " " << lon << " " << alt << " " << roll <<  " " << pitch <<  " " << yaw <<  " " \
                << vn <<  " " << ve <<  " " << vf <<  " " << vl <<  " " << vu <<  " " << ax <<  " " << ay << " " \
                << az <<  " " << af <<  " " << al <<  " " << au <<  " " << wx <<  " " << wy <<  " " << wz << " " \
                << wf <<  " " << wl <<  " " << wu <<  " " << pos_accuracy <<  " " << vel_accuracy <<  " " \
                << navstat <<  " " << numsats << " " << posmode <<  " " << velmode <<  " " << orimode <<  endl;
  poseKittiFile.close();

  // TODO(farzad) Add support for TUM dataset

  // TUM RGB-D dataset trajectory format:
  // Every row has 8 entries containing timestamp (in seconds), 
  // position and orientation (as quaternion) with each value separated by a space:
  // timestamp x y z q_x q_y q_z q_w
  // std::ofstream poseTumFile;
  // poseTumFile.open(poseTUMFrameEntry, std::ios::app)
  // poseTumFile << fix_msg->header.stamp.toNSec() << " " << 
  
  string poseTimestamp = formatTimestamp(fix_msg->header.stamp.sec, fix_msg->header.stamp.nsec);
  poseTimestampFile << poseTimestamp << endl;

  kitti::Transformation pose_map;
  std::vector<double> pose_double {lat, lon, alt, roll, pitch, yaw};
  parser_.convertGpsToPose(pose_double, &pose_map);
  Eigen::Matrix4d pose_map_mat = pose_map.getTransformationMatrix();

  if (current_entry_ == 0) {
    current_timestamp_ns_ = base_timestamp_ns_ = fix_msg->header.stamp.toNSec();
  }else{
    publish_dt_ns_ = fix_msg->header.stamp.toNSec() - base_timestamp_ns_;
    current_timestamp_ns_ = fix_msg->header.stamp.toNSec();
  }

  if(isRectified && !isCalibDone){
    tf::Transform P2 = tf::Transform(toTFMat(cameraMatrixRectified), tf::Vector3(0, 0, 0));
    cout << "P2: " << fmtTransToKittiCalib(P2) << endl;

    // Publish Raw/odom calib files only once!
    float* S_rect_02 = new float[2]{(float) imageSizeRectified.width, (float) imageSizeRectified.height}; 
    writeRawCalibFiles(S_02, toTFMat(cameraMatrix), D_02, S_rect_02, P2);

    std::string odometryCalibPath = odometrySeqXFolder + "/" + "calib.txt";
    writeOdometryCalib(odometryCalibPath, P2);
    isCalibDone = true;
  }
  
  string dt_sec = (boost::format("%e") % ((double) publish_dt_ns_ * 1e-9)).str();
  std::cout << "dt (sec): " << dt_sec << std::endl;

  cout << "------------- Relative Pose -------------" << endl;
  cout << pose_map_mat.block<3,4>(0, 0) << endl;
  
  // Publish processed odometry poses.
  odometryPosesFile << pose_map_mat.block<3,4>(0, 0).format(*commaInitFmt) << endl;
  odometryTimesFile << dt_sec << endl;

  // Publish detection calib files.
  tf::Transform P2 = tf::Transform(toTFMat(cameraMatrixRectified), tf::Vector3(0, 0, 0));
  std::string detectionCalibPath = detectionCalibFolder + "/" + baseFileName + ".txt";
  writeDetectionCalibs(detectionCalibPath, P2);

  // // Check longest sync gap
  // image_msg->header.stamp;
  // pc_msg->header.stamp;
  // fix_msg->header.stamp;
  // imu_msg->header.stamp
  // std::sort
  current_entry_++;
  return true;

}

void KittiLiveNode::undistortImage(const cv::Mat& image, cv::Mat& image_undistort,
                                   cv::Mat cameraMatrix, cv::Mat distorsionCoeff,
                                   cv::Size imageSize, cv::Mat& newCameraMatrix, cv::Size& newImageSize){
  
  cv::Rect validPixROI;
  cv::Mat outputImage;
  newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distorsionCoeff, imageSize, 1, imageSize, &validPixROI);
  cv::undistort(image, outputImage, cameraMatrix, distorsionCoeff, newCameraMatrix);
  outputImage(validPixROI).copyTo(image_undistort);
  newImageSize = cv::Size(image_undistort.cols, image_undistort.rows);
  isRectified = true;
  // cv::imshow("Undistort copped image", image_undistort);
  // cv::waitKey(0);
}
string KittiLiveNode::formatTimestamp(uint32_t timestamp_s, uint32_t timestamp_ns){
  seconds sec = seconds(timestamp_s); 
  system_clock::time_point tp{sec};
  time_t ts_time_t = system_clock::to_time_t(tp);

  char buffer[80];
  strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S", localtime(&ts_time_t));
  
  std::string ts_sec(buffer);
  std::string ts_nsec = (boost::format(".%09u") % timestamp_ns).str();
  return ts_sec + ts_nsec;
}

bool KittiLiveNode::checkAndCreateFile(std::string path, std::ofstream& file){
  if (std::ifstream(path)){
    cout << "File already exists: " << path << std::endl;
    return false;
  }
  file.open(path);
  if (!file)
  {
      std::cout << "File could not be created: " << path << std::endl;
      return false;
  }
  return true;
}

string KittiLiveNode::formatMatrix3x3(const tf::Matrix3x3& matrix3x3){
  char buff[256];
  sprintf(buff, "%e %e %e %e %e %e %e %e %e", matrix3x3[0][0], matrix3x3[0][1], matrix3x3[0][2],
                                              matrix3x3[1][0], matrix3x3[1][1], matrix3x3[1][2],
                                              matrix3x3[2][0], matrix3x3[2][1], matrix3x3[2][2]);
  string strMat = string(buff);
  return strMat;
}

tf::Matrix3x3 KittiLiveNode::toTFMat(const cv::Mat mat){

  tf::Matrix3x3 tf_mat(mat.at<double>(0, 0), mat.at<double>(0, 1), mat.at<double>(0, 2),
                       mat.at<double>(1, 0), mat.at<double>(1, 1), mat.at<double>(1, 2),
                       mat.at<double>(2, 0), mat.at<double>(2, 1), mat.at<double>(2, 2));
  return tf_mat;
}

string KittiLiveNode::formatVector3(const tf::Vector3& vector3){
  char buff[128];
  sprintf(buff, "%e %e %e", vector3[0], vector3[1], vector3[2]);
  return string(buff);
}

string KittiLiveNode::fmtTransToKittiCalib(tf::Transform transform)
{
  tf::Vector3 origin = transform.getOrigin();
  tf::Matrix3x3 rotation_mat = transform.getBasis();
  auto strFmt = boost::format("%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f \n") % rotation_mat[0].getX() % rotation_mat[0].getY() % rotation_mat[0].getZ() % origin.getX() \
                                                                                  % rotation_mat[1].getX() % rotation_mat[1].getY() % rotation_mat[1].getZ() % origin.getY() \
                                                                                  % rotation_mat[2].getX() % rotation_mat[2].getY() % rotation_mat[2].getZ() % origin.getZ();
  return strFmt.str();
}

/**
 * @brief 
 * 
 * @param S_02 1x2 size of image xx before rectification
 * @param K_02 calibration matrix of camera xx before rectification
 * @param D_02 1x5 distortion vector of camera xx before rectification
 * @param S_rect_02 image size after rectificatio
 * @param P_rect_02 projection matrix after rectification
 * @param R_02 3x3 rotation matrix of camera xx (extrinsic)
 * @param T_02 3x1 translation vector of camera xx (extrinsic)
 * @param R_rect_02 3x3 rectifying rotation to make image planes co-planar
 */
void KittiLiveNode::writeRawCalibFiles(float* S_02, tf::Matrix3x3 K_02, float* D_02, float* S_rect_02, tf::Transform P_rect_02)
{ 

  tf::Matrix3x3 R_02 = tf::Matrix3x3::getIdentity();
  tf::Vector3 T_02 = tf::Vector3(0, 0, 0);
  tf::Matrix3x3 R_rect_02 = tf::Matrix3x3::getIdentity();

  // TODO(farzad) P_rect_02 has different values than the P_02 in Object detection calib file. Find out why?!

  // Write calib_cam_to_cam.txt --------------------------------------------------
  std::ofstream calib_cam_to_cam;
  std::string calib_cam_to_cam_path =  dataset_path_ + "/raw/" + "calib_cam_to_cam.txt";
  if (!checkAndCreateFile(calib_cam_to_cam_path, calib_cam_to_cam)) return;
  
  calib_cam_to_cam << "calib_time: " << calib_time_ <<  endl;
  calib_cam_to_cam << "corner_dist: 0.0" <<  endl;
  
  string s_02 = (boost::format("S_02: %e %e") % S_02[0] % S_02[1]).str();
  string s_rect_02 = (boost::format("S_rect_02: %e %e") % S_rect_02[0] % S_rect_02[1]).str();
  string d_02 = (boost::format("D_02: %e %e %e %e %e") % D_02[0] % D_02[1] % D_02[2] % D_02[3] % D_02[4]).str();

  calib_cam_to_cam << s_02 << endl;
  calib_cam_to_cam << "K_02:" << formatMatrix3x3(K_02) << endl;
  calib_cam_to_cam << d_02 << endl;
  calib_cam_to_cam << "R_02: " << formatMatrix3x3(R_02) << endl;
  calib_cam_to_cam << "T_02: " << formatVector3(T_02) << endl;
  calib_cam_to_cam << s_rect_02 << endl;
  calib_cam_to_cam << "R_rect_02: " << formatMatrix3x3(R_rect_02) << endl;
  calib_cam_to_cam << "P_rect_02: " << fmtTransToKittiCalib(P_rect_02) << endl;
  calib_cam_to_cam.close();

  // // Write calib_cam_to_cam.txt --------------------------------------------------
  std::ofstream calib_velo_to_cam;
  std::string calib_velo_to_cam_path =  dataset_path_ + "/raw/" + "calib_velo_to_cam.txt";
  if (!checkAndCreateFile(calib_velo_to_cam_path, calib_velo_to_cam)) return;
  calib_velo_to_cam << "calib_time: " << calib_time_ <<  endl;
  calib_velo_to_cam << "R: " << formatMatrix3x3(velo2CamTrnasform_.getBasis()) <<  endl;
  calib_velo_to_cam << "T: " << formatVector3(velo2CamTrnasform_.getOrigin()) <<  endl;
  calib_velo_to_cam << "delta_f: 0, 0"  <<  endl;
  calib_velo_to_cam << "delta_c: 0, 0"  <<  endl;
  calib_velo_to_cam.close();
  
  // // Write calib_imu_to_velo.txt --------------------------------------------------
  std::ofstream calib_imu_to_velo;
  std::string calib_imu_to_velo_path =  dataset_path_ + "/raw/" + "calib_imu_to_velo.txt";
  if (!checkAndCreateFile(calib_imu_to_velo_path, calib_imu_to_velo)) return;
  calib_imu_to_velo << "calib_time: " << calib_time_ <<  endl;
  calib_imu_to_velo << "R: " << formatMatrix3x3(velo2ImuTrnasform_.inverse().getBasis()) <<  endl;
  calib_imu_to_velo << "T: " << formatVector3(velo2ImuTrnasform_.inverse().getOrigin()) <<  endl;
  calib_imu_to_velo.close();
  
  // Write calib_gps_to_velo.txt --------------------------------------------------
  std::ofstream calib_gps_to_velo;
  std::string calib_gps_to_velo_path =  dataset_path_ + "/raw/" + "calib_gps_to_velo.txt";
  if (!checkAndCreateFile(calib_gps_to_velo_path, calib_gps_to_velo)) return;
  calib_gps_to_velo << "calib_time: " << calib_time_ <<  endl;
  calib_gps_to_velo << "R: " << formatMatrix3x3(velo2GpsTransform_.inverse().getBasis()) <<  endl;
  calib_gps_to_velo << "T: " << formatVector3(velo2GpsTransform_.inverse().getOrigin()) <<  endl;
  calib_gps_to_velo.close();
}

void KittiLiveNode::startPublishing(double rate_hz) {
  double publish_dt_sec = 1.0 / rate_hz;
  publish_dt_ns_ = static_cast<uint64_t>(publish_dt_sec * 1e9);
  std::cout << "Publish dt ns: " << publish_dt_ns_ << std::endl;
  publish_timer_ = nh_.createWallTimer(ros::WallDuration(publish_dt_sec),
                                       &KittiLiveNode::timerCallback, this);
}

void KittiLiveNode::timerCallback(const ros::WallTimerEvent& event) {
  Transformation tf_interpolated;

  std::cout << "Current entry: " << current_entry_ << std::endl;

  if (current_entry_ == 0) {
    // This is the first time this is running! Initialize the current timestamp
    // and publish this entry.
    if (!publishEntry(current_entry_)) {
      publish_timer_.stop();
    }
    current_timestamp_ns_ = parser_.getPoseTimestampAtEntry(current_entry_);
    publishClock(current_timestamp_ns_);
    if (parser_.interpolatePoseAtTimestamp(current_timestamp_ns_,
                                           &tf_interpolated)) {
      publishTf(current_timestamp_ns_, tf_interpolated);
    }
    current_entry_++;
    return;
  }

  std::cout << "Publish dt ns: " << publish_dt_ns_ << std::endl;
  current_timestamp_ns_ += publish_dt_ns_;
  std::cout << "Updated timestmap: " << current_timestamp_ns_ << std::endl;
  publishClock(current_timestamp_ns_);
  if (parser_.interpolatePoseAtTimestamp(current_timestamp_ns_,
                                         &tf_interpolated)) {
    publishTf(current_timestamp_ns_, tf_interpolated);
    // std::cout << "Transform: " << tf_interpolated << std::endl;
  } else {
    std::cout << "Failed to interpolate!\n";
  }

  std::cout << "Current entry's timestamp: "
            << parser_.getPoseTimestampAtEntry(current_entry_) << std::endl;
  if (parser_.getPoseTimestampAtEntry(current_entry_) <=
      current_timestamp_ns_) {
    if (!publishEntry(current_entry_)) {
      publish_timer_.stop();
      return;
    }
    current_entry_++;
  }
}

void KittiLiveNode::publishClock(uint64_t timestamp_ns) {
  ros::Time timestamp_ros;
  timestampToRos(timestamp_ns, &timestamp_ros);
  rosgraph_msgs::Clock clock_time;
  clock_time.clock = timestamp_ros;
  clock_pub_.publish(clock_time);
}

bool KittiLiveNode::publishEntry(uint64_t entry) {
  ros::Time timestamp_ros;
  uint64_t timestamp_ns;
  rosgraph_msgs::Clock clock_time;

  // Publish poses + TF transforms + clock.
  Transformation pose;
  if (parser_.getPoseAtEntry(entry, &timestamp_ns, &pose)) {
    geometry_msgs::PoseStamped pose_msg;
    geometry_msgs::TransformStamped transform_msg;

    timestampToRos(timestamp_ns, &timestamp_ros);
    pose_msg.header.frame_id = world_frame_id_;
    pose_msg.header.stamp = timestamp_ros;
    transform_msg.header.frame_id = world_frame_id_;
    transform_msg.header.stamp = timestamp_ros;

    poseToRos(pose, &pose_msg);
    transformToRos(pose, &transform_msg);

    pose_pub_.publish(pose_msg);
    transform_pub_.publish(transform_msg);

    // publishClock(timestamp_ns);
    // publishTf(timestamp_ns, pose);
  } else {
    return false;
  }

  // Publish images.
  cv::Mat image;
  for (size_t cam_id = 0; cam_id < parser_.getNumCameras(); ++cam_id) {
    if (parser_.getImageAtEntry(entry, cam_id, &timestamp_ns, &image)) {
      sensor_msgs::Image image_msg;
      imageToRos(image, &image_msg);

      // TODO(helenol): cache this.
      // Get the calibration info for this camera.
      CameraCalibration cam_calib;
      parser_.getCameraCalibration(cam_id, &cam_calib);
      sensor_msgs::CameraInfo cam_info;
      calibrationToRos(cam_id, cam_calib, &cam_info);

      timestampToRos(timestamp_ns, &timestamp_ros);

      image_msg.header.stamp = timestamp_ros;
      image_msg.header.frame_id = getCameraFrameId(cam_id);
      cam_info.header = image_msg.header;

      image_pubs_[cam_id].publish(image_msg, cam_info, timestamp_ros);
    }
  }

  // Publish pointclouds.
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  if (parser_.getPointcloudAtEntry(entry, &timestamp_ns, &pointcloud)) {
    timestampToRos(timestamp_ns, &timestamp_ros);

    // This value is in MICROSECONDS, not nanoseconds.
    pointcloud.header.stamp = timestamp_ns / 1000;
    pointcloud.header.frame_id = velodyne_frame_id_;
    pointcloud_pub_.publish(pointcloud);
  }

  return true;
}

void KittiLiveNode::publishTf(uint64_t timestamp_ns,
                              const Transformation& imu_pose) {
  ros::Time timestamp_ros;
  timestampToRos(timestamp_ns, &timestamp_ros);
  Transformation T_imu_world = imu_pose;
  Transformation T_vel_imu = parser_.T_vel_imu();
  Transformation T_cam_imu;

  tf::Transform tf_imu_world, tf_cam_imu, tf_vel_imu;

  transformToTf(T_imu_world, &tf_imu_world);
  transformToTf(T_vel_imu.inverse(), &tf_vel_imu);

  tf_broadcaster_.sendTransform(tf::StampedTransform(
      tf_imu_world, timestamp_ros, world_frame_id_, imu_frame_id_));
  tf_broadcaster_.sendTransform(tf::StampedTransform(
      tf_vel_imu, timestamp_ros, imu_frame_id_, velodyne_frame_id_));

  for (size_t cam_id = 0; cam_id < parser_.getNumCameras(); ++cam_id) {
    T_cam_imu = parser_.T_camN_imu(cam_id);
    transformToTf(T_cam_imu.inverse(), &tf_cam_imu);
    tf_broadcaster_.sendTransform(tf::StampedTransform(
        tf_cam_imu, timestamp_ros, imu_frame_id_, getCameraFrameId(cam_id)));
  }
}

}  // namespace kitti

int main(int argc, char** argv) {
  ros::init(argc, argv, "kitti_live");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  if (argc < 3) {
    std::cout << "Usage: rosrun kitti_to_rosbag kitti_live_node "
                 "calibration_path dataset_path\n";
    std::cout << "Note: no trailing slashes.\n";
    return 0;
  }

  const std::string calibration_path = argv[1];
  const std::string dataset_path = argv[2];
   
  kitti::KittiLiveNode node(nh, nh_private, calibration_path, dataset_path);

  // node.startPublishing(50.0);

  ros::spin();

  return 0;
}
