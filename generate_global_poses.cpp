// read in tag id, timestamp (us), pose R, t and output the true pose
// information
// verices: tag poses and robot poses
// edges: 1. tag-pose edge, between robot pose and tag pose edge  2. pose-pose
// edge, between two robot poses

#define G2O
#include <fstream>
#include <iostream>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <sstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#ifdef G2O
#include "g2o/core/factory.h"
#include "g2o/stuff/command_args.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#endif
// #include <opencv2/viz.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
#ifdef G2O
using namespace g2o;
#endif
using namespace cv;

string sDate = "2016-12-08_17-10-33";
#ifdef _WIN64
string aprilTag_base = "D:\\Work\\data\\AprilTagData\\AprilTag\\";
string aprilTag_outPath = aprilTag_base + sDate + "\\";
string platform_base =
    "D:\\Work\\data\\AprilTagData\\tabletRecordedImages\\" + sDate + "\\";
#else
string aprilTag_base = "/home/test1/Documents/TagSlam/apriltags1/output/";
string aprilTag_outPath = aprilTag_base + sDate + "/";
string platform_base =
    "/home/test1/Documents/TagSlam/apriltags1/tabletRecordedImages/" + sDate +
    "/";
#endif
string file_out_observations = aprilTag_outPath + "observations.dat";
string file_recorded_path =
    aprilTag_base + sDate + ".dat"; // tagid detected, timestamp, robot rotation

// matrix, robot translation
string file_out_cam_pose = aprilTag_outPath + "cam_pose.dat";
string file_out_optimized_cam_pose =
    aprilTag_outPath + "optimized_cam_pose.dat";
string file_out_optimized_tag_pose =
    aprilTag_outPath + "optimized_tag_pose.dat";
string file_out_short_cam_poses = aprilTag_outPath + "short_cam_poses.dat";

string file_head_pose = platform_base + "headpose.txt";
string file_odom = platform_base + "odom.txt";
string file_body_pose = platform_base + "basepose.txt";

// common for two cams
const double global_body_z = 0;
const double body_local_x = 0;
const double body_local_y = 0;
const double body_local_z = 0.14789;

// platform cam
const double head_cam_x = 0.04288;
const double head_cam_y = 0.05434;
const double head_cam_z = 0.01822;
const double body_head_x = 0.01346;
const double body_head_y = 0;
const double body_head_z = 0.46811; // in meter assume body z is the ground

// ds4color cam
const double body_cam_x = 0.05634;
const double body_cam_y = -0.04195;
const double body_cam_z = 0.39911;

int N = 1;

#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0 * PI;

Mat test_image = Mat::zeros(720, 1440, CV_8UC3);

struct Observation {
  int timestamp;
  map<int, Eigen::Isometry3d> map_tagid_reltransform;
};

int readInObservationAndTagID(const char *file,
                              vector<Observation> &vector_observation,
                              vector<int> &vector_tag);
int readInEstimatedGlobalRobotPose(
    const char *file_odom, const char *file_head_pose,
    map<int, Eigen::Isometry3d> &map_timestamp_robotpose);
#ifdef G2O
void writeToG2o(vector<VertexSE3 *> &vertices, vector<EdgeSE3 *> &edges,
                string out_file_name);
#endif
void wRo_to_euler(const Eigen::Matrix3d &wRo, double &yaw, double &pitch,
                  double &roll);
inline double standardRad(double t);
void writeRotationMatrixToAngles(Eigen::Matrix3d rot, Eigen::Vector3d trans,
                                 ofstream &outfile);

int main(int argc, char **argv) {
  // command line parsing
  double radius;
  string out_file_name;
#ifdef G2O
  CommandArgs arg;
  arg.param("o", out_file_name, "-", "output filename");
  arg.param("radius", radius, 100., "radius of the sphere");
  arg.parseArgs(argc, argv);
#endif
  vector<Observation> observations;
  vector<int> tagids;

  map<int, Eigen::Isometry3d> map_timestamp_robotpose;

  ofstream outfile_optimized_cam_pose(file_out_optimized_cam_pose, ios::out);
  ofstream outfile_optimized_tag_pose(file_out_optimized_tag_pose, ios::out);
  ofstream outfile_short_cam_poses(file_out_short_cam_poses, ios::out);
  ofstream outfile_observations(file_out_observations, ios::out);

  if (!outfile_optimized_cam_pose || !outfile_optimized_tag_pose ||
      !outfile_short_cam_poses) {
    cerr << "open outfile error" << endl;
    exit(1);
  }

  const char *file_recorded_path_ = file_recorded_path.c_str();
  const char *file_odom_ = file_odom.c_str();
  const char *file_head_pose_ = file_head_pose.c_str();
  if (!readInObservationAndTagID(file_recorded_path_, observations, tagids))
    return 0;
  else if (!readInEstimatedGlobalRobotPose(file_odom_, file_head_pose_,
                                           map_timestamp_robotpose))
    return 0;
  else {
    // tagids.clear();
    // tagids.push_back(1);
    // tagids.push_back(2);
    // tagids.push_back(3);
    // tagids.push_back(4);
    cout << "In main, got " << observations.size() << " observations" << endl;
#ifdef G2O
    // create vertices and edges (graphic method starts here)
    vector<VertexSE3 *> vertices;
    vector<EdgeSE3 *> edges;
    vector<EdgeSE3 *> odometryEdges; // odometry edge is assumed to connect
                                     // vertices whose id only differs by one

    // declare optimizer
    SparseOptimizer optimizer;
    // initialize Cholmod linear optimizer for sparse data
    BlockSolver_6_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    // 6*3 ??
    BlockSolver_6_3 *block_solver = new g2o::BlockSolver_6_3(linearSolver);
    // L-M ??
    OptimizationAlgorithmLevenberg *algorithm =
        new g2o::OptimizationAlgorithmLevenberg(block_solver);

    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);
#endif
    vector<Eigen::Isometry3d> T_cams;
    vector<int> vector_int_to_remove;
    int prev_accepted_timestamp = 0;

    // get T_cams from odometry, matching timestamp between observation and
    // odometry
    for (int i = 0; i < observations.size(); i++) // debug
    {
      int t = observations[i].timestamp;
      map<int, Eigen::Isometry3d>::iterator it_low, it_high, it;

      int low = t - 20000;
      int high = t + 20000;
      // cout << "t: " << t << " ";

      if (low <= prev_accepted_timestamp)
        low = prev_accepted_timestamp + 50;

      it_low = map_timestamp_robotpose.lower_bound(low);
      it_high = map_timestamp_robotpose.upper_bound(high);

      // cout << it_low->first << " ";
      // cout << it_high->first << " ";

      if (it_low->first > it_high->first) {
        // delete the entry from observations and move on
        vector_int_to_remove.push_back(t);
      } else if (it_low->first == it_high->first) {
        T_cams.push_back(map_timestamp_robotpose[it_low->first]);

        prev_accepted_timestamp = it_low->first;

        // cout << it_low->first << " ";
      } else {
        // initialize t_closest
        int t_closest = it_low->first;
        for (map<int, Eigen::Isometry3d>::iterator it = it_low; it != it_high;
             it++) {
          if (abs(t - t_closest) > abs(t - it->first)) {
            t_closest = it->first;
          }
        }

        T_cams.push_back(map_timestamp_robotpose[t_closest]);
        prev_accepted_timestamp = t_closest;

        // cout << (map_timestamp_robotpose[t_closest]).matrix() << endl;
      }
    }

    // now remove the observation entries that can't find matching timestamps
    for (int i = 0; i < vector_int_to_remove.size(); i++) {
      for (vector<Observation>::iterator it = observations.begin();
           it != observations.end(); it++) {
        if ((*it).timestamp == vector_int_to_remove[i]) { // found
          // erase
          observations.erase(it);
          break;
          cout << "erase bad timestamp match entries" << endl;
        }
      }
    }

    // check values
    cout << "size of observations " << observations.size() << "size of T_cams "
         << T_cams.size() << endl;
    // vector < Observation >::iterator it;
    // vector < Eigen::Vector3d >::iterator T_cams = T_cams.begin();

    // int max_difference = 0;
    for (int i = 0; i < observations.size() - 1; i++) {
      // cout << observations[i].timestamp << endl;
      Point2d pt((observations[i].timestamp % 100000) * 0.05, 200);
      circle(test_image, pt, 2, Scalar(0, 0, 255), 2);

      // max_difference = max(-observations[i].timestamp +
      // observations[i+1].timestamp, max_difference);
      // cout << i << " " << observations[i].timestamp << " " << max_difference
      // << endl;
    }

    // erase the first value, because somehow the first value of T_robot is very
    // wrong
    observations.erase(observations.begin());
    T_cams.erase(T_cams.begin());

    // check values before putting in vertices
    // for (int i = 0; i< T_cams.size(); i++){
    //    cout << "T_robot" << endl << T_cams[i].matrix() << endl;
    // }

    int found = 0;
    // add all tag poses to vertices, with estimated pose initialization
    for (int i_tag = 0; i_tag < tagids.size(); i_tag++) {
      Eigen::Isometry3d T_tag, T_robot, T_tag_relative_robot;
      int i_observation = 0;
#ifdef G2O
      VertexSE3 *v = new VertexSE3;
      v->setId(i_tag);
#endif
      found = 0;

      // loop the observation vector to find one T_tag_relative_robot and the
      // respective T_robot for this tag
      // while (found < observations.size()-3 && i_observation <
      // observations.size())
      while (found == 0 && i_observation < observations.size()) {
        // cout << "here" << endl;
        map<int, Eigen::Isometry3d> m =
            observations[i_observation].map_tagid_reltransform;
        // cout << "current tag ID to find " << i_tag << " | in the map is tag
        // ID"<< m.begin()->first << endl;

        // loop each observation to find tag id
        if (m.count(tagids[i_tag])) {
          T_tag_relative_robot = m[tagids[i_tag]];

          T_robot = T_cams[i_observation];

          // VertexSE3* v = new VertexSE3;
          // v->setId(found);

          T_tag = T_robot * T_tag_relative_robot; // local transformation from
                                                  // robot to tag (multiply on
                                                  // the right)
#ifdef G2O
          // get a good initialization of tag pose from robot pose, using the
          // observed relative pose we got from april tag
          v->setEstimate(T_tag);
          optimizer.addVertex(v);
          vertices.push_back(v);
#endif
          found = 1;

          writeRotationMatrixToAngles(
              T_tag_relative_robot.matrix().block<3, 3>(0, 0),
              T_tag_relative_robot.translation(), outfile_observations);

          // found ++;

          // cout << "T_tag " << endl << T_tag.matrix() << endl;
        }
        // cout << "found " << found << endl;
        i_observation++;
      }
    }

    int count = 0;
    // add all robot poses to vertices
    for (int i = 0; i < T_cams.size(); i += N) {
#ifdef G2O
      VertexSE3 *v = new VertexSE3();

      v->setId(tagids.size() + count);
      // v->setId(i);
      // v->setId(i + found);
      v->setEstimate(
          T_cams[count]); // SE3Quat(const Matrix3D& R, const Vector3D& t)
      optimizer.addVertex(v);
      vertices.push_back(v);
#endif
      Eigen::Matrix3d rot = T_cams[count].matrix().block<3, 3>(0, 0);
      Eigen::Vector3d trans = T_cams[count].translation();
      writeRotationMatrixToAngles(rot, trans, outfile_short_cam_poses);

      count++;
    }
    cout << "added robot poses to vertices" << count << endl;
#ifdef G2O
    // generate tag-robot edges, for each pose if there is a measurement to a
    // tag, there is an edge to that tag
    for (int i = 0; i < vertices.size() - tagids.size(); i++) {
      // for (int j = 0; j < found; j++)
      for (int j = 0; j < tagids.size(); j++) {
        cout << "tagids___" << tagids.size() << " " << j << " ";
        // if (observations[i].map_tagid_reltransform.count(tagids[0]))
        if (observations[i].map_tagid_reltransform.count(tagids[j])) {
          // cout << j << " ";
          map<int, Eigen::Isometry3d> m =
              observations[i].map_tagid_reltransform;

          EdgeSE3 *e = new EdgeSE3;

          // e->setVertex(0, vertices[i + found] );
          e->setVertex(
              0, vertices[tagids.size() + i]); // 0: from  1 is the fixed point
          e->setVertex(1, vertices[j]);
          e->setMeasurement(m[tagids[j]]);
          cout << "check observations in main:" << endl
               << (m[tagids[j]]).matrix() << endl;
          Eigen::Matrix<double, 6, 6> info;
          info = Eigen::Matrix<double, 6, 6>::Identity();
          cout << "info before" << endl;
          info << 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.3, 0,
              0, 0, 1;
          cout << "info" << endl;
          e->setInformation(
              info); // const information matrix to-do identity matrix???

          // core function
          e->setRobustKernel(new g2o::RobustKernelHuber());
          optimizer.addEdge(e);
          edges.push_back(e);

          cout << "Edge: robot pose " << i << " to tag " << j << endl;
        }
      }
    }

    // create odometry edges
    for (size_t i = tagids.size(); i < vertices.size(); ++i) {
      VertexSE3 *prev = vertices[i - 1]; // create vetex
      VertexSE3 *cur = vertices[i];
      Eigen::Isometry3d t = prev->estimate().inverse() * cur->estimate();
      EdgeSE3 *e = new EdgeSE3;
      Eigen::Matrix<double, 6, 6> info;
      // info <<
      // 0.5, 0.5, 0.5, 1;
      // 0.5, 0.5, 0.5, 1;
      // 0.5, 0.5, 0.5, 1;
      // 0, 0, 0, 1;
      // info = Eigen::Matrix<double, 6, 6>::Identity();
      e->setVertex(0, prev);
      e->setVertex(1, cur);
      e->setMeasurement(t);
      e->setInformation(info);
      odometryEdges.push_back(e);
      edges.push_back(e);
    }

    cout << "Optimization Starts" << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(5000);
    cout << "Optimization Ends" << endl;

    // optimized pose of the robot
    for (size_t i = 0; i < observations.size() - 20; i++) {
      VertexSE3 *v = dynamic_cast<VertexSE3 *>(
          optimizer.vertex(tagids.size() + i)); // why it fails when i+=N ?
      Eigen::Isometry3d pose = v->estimate();

      Eigen::Vector3d trans = pose.translation();
      Eigen::Matrix3d rot = pose.matrix().block<3, 3>(0, 0);

      writeRotationMatrixToAngles(rot, trans, outfile_optimized_cam_pose);
    }

    for (size_t i = 0; i < tagids.size(); i++) {
      cout << "tagids" << tagids.size() << endl;
      VertexSE3 *v = dynamic_cast<VertexSE3 *>(optimizer.vertex(i));
      Eigen::Isometry3d pose = v->estimate();

      // cout<<pose.matrix() <<endl;
      Eigen::Matrix3d rot = pose.matrix().block<3, 3>(0, 0);
      Eigen::Vector3d trans = pose.translation();
      writeRotationMatrixToAngles(rot, trans, outfile_optimized_tag_pose);
      // cout << i << endl;
    }
    cout << "tagids" << tagids.size() << endl;

    writeToG2o(vertices, edges, out_file_name);
#endif
  }
  return 0;
}

// get a reference of robot pose from odometry and head pos
int readInEstimatedGlobalRobotPose(
    const char *file_odom, const char *file_head_pose,
    map<int, Eigen::Isometry3d> &map_timestamp_robotpose) {
  ofstream outfile_cam_poses(file_out_cam_pose, ios::out);

  if (!outfile_cam_poses) {
    cerr << "open outfile_cam_poses error" << endl;
    exit(1);
  }

  string line_odom, line_head_pose, line_body_pose;
  ifstream rfile_odom(file_odom);
  ifstream rfile_head_pose(file_head_pose);
  ifstream rfile_body_pose(file_body_pose);

  if (rfile_odom.is_open() && rfile_head_pose.is_open() &&
      rfile_body_pose.is_open()) {
    while (getline(rfile_odom, line_odom) &&
           getline(rfile_head_pose, line_head_pose) &&
           getline(rfile_body_pose, line_body_pose)) {
      istringstream iss_odom(line_odom);
      istringstream iss_head_pose(line_head_pose);
      istringstream iss_body_pose(line_body_pose);

      vector<string> tmp;

      copy(istream_iterator<string>(iss_odom), istream_iterator<string>(),
           back_inserter(tmp));
      copy(istream_iterator<string>(iss_head_pose), istream_iterator<string>(),
           back_inserter(tmp));
      copy(istream_iterator<string>(iss_body_pose), istream_iterator<string>(),
           back_inserter(tmp));

      Eigen::Matrix3d rot;
      Eigen::Isometry3d T_tag_relative_robot, T_cam_pose;
      Eigen::Isometry3d T_body_head, T_head_cam, T_global_body, T_body_local;

      int timestamp_odom, timestamp_head_pose;
      double pos_x, pos_y, body_yaw, head_pitch, head_yaw, body_pitch;

      for (int i = 0; i < tmp.size(); i++) {
        // cout << i << " " << tmp[i].c_str() << endl;

        if (i == 0) {
          timestamp_odom = atof(tmp[i].c_str());
        }

        else if (i == 1) {
          pos_x = atof(tmp[i].c_str());
        }

        else if (i == 2) {
          pos_y = atof(tmp[i].c_str());
        } // to-do

        else if (i == 3) {
          body_yaw = atof(tmp[i].c_str());
        }

        else if (i == 8) {
          timestamp_head_pose = atof(tmp[i].c_str());
        }

        else if (i == 9) {
          head_pitch = atof(tmp[i].c_str());
        } // head_pitch = robot_pitch

        else if (i == 11) {
          head_yaw = atof(tmp[i].c_str());
        }

        else if (i == 15) {
          body_pitch = 2 * atof(tmp[i].c_str());
        }
      }

      if (abs(timestamp_odom - timestamp_head_pose) > 100000)
        cout << "odom and headpose timestamp differ more than 100ms " << endl;

      Eigen::Matrix3d R_body_head_z;
      R_body_head_z << cos(head_yaw), -sin(head_yaw), 0, sin(head_yaw),
          cos(head_yaw), 0, 0, 0, 1;

      Eigen::Matrix3d R_body_head_y;
      R_body_head_y << cos(head_pitch), 0, -sin(head_pitch), 0, 1, 0,
          sin(head_pitch), 0, cos(head_pitch);

      Eigen::Matrix3d R_global_body_z;
      R_global_body_z << cos(body_yaw), -sin(body_yaw), 0, sin(body_yaw),
          cos(body_yaw), 0, 0, 0, 1;

      Eigen::Matrix3d R_body_local;
      // cout << "body pitch" << body_pitch << endl;
      R_body_local << // y
          cos(body_pitch),
          0, -sin(body_pitch), 0, 1, 0, sin(body_pitch), 0, cos(body_pitch);

      Eigen::Matrix3d R_head_cam = Eigen::Matrix3d::Identity();

      T_head_cam = R_head_cam;
      T_head_cam.translation() << head_cam_x, head_cam_y, head_cam_z;
      // cout << "T_head_cam: " << endl << T_head_cam.matrix() << endl;

      T_body_head = R_body_head_z * R_body_head_y;
      T_body_head.translation() << body_head_x, body_head_y, body_head_z;
      // cout << "T_body_head:" << endl << T_body_head.matrix() << endl;

      T_global_body = R_global_body_z;
      T_global_body.translation() << pos_x, pos_y, global_body_z;
      // cout << "T_global_body: " << endl << T_global_body.matrix() << endl;

      T_body_local = R_body_local;
      T_body_local.translation() << body_local_x, body_local_y, body_local_z;

      // platform camera
      // T_cam_pose = T_global_body * T_body_local * T_body_head * T_head_cam;

      // ds4color camera
      Eigen::Isometry3d T_body_cam;
      T_body_cam = Eigen::Matrix3d::Identity();
      T_body_cam.translation() << body_cam_x, body_cam_y, body_cam_z;
      T_cam_pose = T_global_body * T_body_local * T_body_cam;

      // cout << "z" << T_cam_pose.translation()(2) << endl;

      map_timestamp_robotpose[timestamp_odom] = T_cam_pose;

      // rot = T_cam_pose.matrix().block<3,3>(0,0);
      // writeRotationMatrixToAngles(rot, T_cam_pose.translation(),
      // outfile_cam_poses);
      // cout << "T_global_cam " << timestamp_odom << endl <<
      // T_cam_pose.translation() << endl;
    }

    rfile_odom.close();
    rfile_head_pose.close();
    return 1;
  } else {
    cout << "Unable to open file";
    return 0;
  }
}

// read current relative pose to an fiducial
int readInObservationAndTagID(
    const char *file, vector<Observation> &vector_observation,
    vector<int> &vector_tag) { // need to point to the input argument address
  string line;

  ifstream rfile(file);
  int count = 0;
  Observation prev_observation;
  int prev_timestamp;

  if (rfile.is_open()) {
    while (getline(rfile, line)) {

      istringstream iss(line);
      vector<string> tmp_observation;

      copy(istream_iterator<string>(iss), istream_iterator<string>(),
           back_inserter(tmp_observation));
      Eigen::Matrix3d rot(3, 3);
      Eigen::Vector3d trans(3);
      Eigen::Isometry3d T_tag_relative_robot;
      int tagID, t;

      for (int i = 0; i < tmp_observation.size(); i++) {

        if (i == 0) {
          tagID = atof(tmp_observation[i].c_str());
        }

        else if (i == 1) {
          t = atof(tmp_observation[i].c_str());
        }

        else if (i >= 2 && i <= 4) {
          trans(i - 2) = atof(tmp_observation[i].c_str());
        }

        else if (i >= 5 && i <= 13) {
          // cout << endl;
          int row = (i - 5) % 3;
          int col = (i - 5) / 3;
          rot(row, col) = (atof(tmp_observation[i].c_str()));
          // cout << row << " " << col << endl;
        }
      }

      trans.norm();

      Eigen::Matrix3d F;
      F << 1, 0, 0, 0, 0, 1, 0, -1, 0;
      Eigen::Matrix3d adjusted_rot = F * rot; // 90 degrees around x
      T_tag_relative_robot =
          rot; // checked with output from jane_get_camera_pose;
      T_tag_relative_robot.translation() = trans;

      // cout << "check observations in readInObservationAndTagID:" << endl <<
      // T_tag_relative_robot.matrix() << endl;

      if (count == 0) {
        prev_observation.map_tagid_reltransform[tagID] = T_tag_relative_robot;
        prev_timestamp = t;

      } // if the timestamp is the same, don't create new entry, add to previous

      if (t == prev_timestamp) {
        // add this observation to the previous tagID~pose map
        prev_observation.map_tagid_reltransform[tagID] = T_tag_relative_robot;

      } else {
        vector_observation.push_back(prev_observation);
        Observation observation;
        int timestamp = t;

        // cout << " Observed tag number : ";
        // for (map <int, Eigen::Isometry3d>::iterator
        // it=prev_observation.map_tagid_reltransform.begin();
        // it!=prev_observation.map_tagid_reltransform.end(); it++)
        // {
        //     cout << it->first << " ";
        // }
        // cout << endl;

        // put this observation into the vector
        observation.timestamp = timestamp;
        observation.map_tagid_reltransform[tagID] = T_tag_relative_robot;

        // update prev observation
        prev_observation = observation;
        prev_timestamp = timestamp;
      }
      count++;

      Eigen::Vector3d trans2 = T_tag_relative_robot.translation();
      // Point2d pt2(640-trans2(0)*100, 320-trans2(1)*100);
      // circle(test_image, pt2, 1, Scalar(255, 255, 0), -1);

      // create a vector of tag ID
      int i = 0;
      bool found = false;
      while (!found && i < vector_tag.size()) // loop vector_tag
      {
        if (vector_tag[i] == tagID)
          found = true;
        i++;
      }
      // !found but i = vector_tag.size()
      if (!found)
        vector_tag.push_back(tagID);
    }
    // cout << "vector_observation " << vector_observation.size() << endl;
    rfile.close();
    return 1;
  } else {
    cout << "Unable to open file";
    return 0;
  }
}

#ifdef G2O
void writeToG2o(vector<VertexSE3 *> &vertices, vector<EdgeSE3 *> &edges,
                string out_file_name) {
  cout << vertices.size() << " helllooooooo" << endl;

  ofstream file_output_stream;
  if (out_file_name != "-") {
    cerr << "Writing into " << out_file_name << endl;
    file_output_stream.open(out_file_name.c_str());
  } else {
    cerr << "writing to stdout" << endl;
  }

  string vertexTag = Factory::instance()->tag(vertices[0]);

  ostream &fout = out_file_name != "-" ? file_output_stream : cout;

  for (size_t i = 0; i < vertices.size(); ++i) {
    VertexSE3 *v = vertices[i];
    fout << vertexTag << " " << v->id() << " ";
    // cout << "v->id " << v->id() << " ";

    v->write(fout);
    fout << endl;
    // cout << i << " " << vertices.size() << endl;
  }

  // cout << endl;

  string edgeTag = Factory::instance()->tag(edges[0]);

  for (size_t i = 0; i < edges.size(); ++i) {
    EdgeSE3 *e = edges[i];
    VertexSE3 *from = static_cast<VertexSE3 *>(e->vertex(0));
    VertexSE3 *to = static_cast<VertexSE3 *>(e->vertex(1));
    fout << edgeTag << " " << from->id() << " " << to->id() << " ";
    // cout << edgeTag << " " << from->id() << " " << to->id() << " ";
    e->write(fout);
    fout << endl;
    // cout << i << " " << from->id() << " " << to->id() << endl;
  }
}
#endif
void writeRotationMatrixToAngles(Eigen::Matrix3d rot, Eigen::Vector3d trans,
                                 ofstream &outfile) {

  // convert to readable angles
  double yaw, pitch, roll;
  wRo_to_euler(rot, yaw, pitch, roll);
  cout << "here" << endl;

  // jane - print degree angles
  double rad2deg = 180.0 / 3.14159;
  yaw = yaw * rad2deg;
  pitch = pitch * rad2deg;
  roll = roll * rad2deg;

  outfile << yaw << " " << pitch << " " << roll << " " << trans(0) << " "
          << trans(1) << " " << trans(2);
  // outfile << trans(0) << " " << trans(1) << " " << trans(2);

  outfile << endl;
}

/**
 * Convert rotation matrix to Euler angles
 */
void wRo_to_euler(const Eigen::Matrix3d &wRo, double &yaw, double &pitch,
                  double &roll) {
  yaw = standardRad(atan2(wRo(1, 0), wRo(0, 0)));
  double c = cos(yaw);
  double s = sin(yaw);
  pitch = standardRad(atan2(-wRo(2, 0), wRo(0, 0) * c + wRo(1, 0) * s));
  roll = standardRad(
      atan2(wRo(0, 2) * s - wRo(1, 2) * c, -wRo(0, 1) * s + wRo(1, 1) * c));
}

/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t + PI, TWOPI) - PI;
  } else {
    t = fmod(t - PI, -TWOPI) + PI;
  }
  return t;
}
