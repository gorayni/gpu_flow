//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Brox et al. [1] and Zach et al. [2] TVL1 Optical Flow
// Dependencies: OpenCV and Qt5 for iterating (sub)directories
// Author: Christoph Feichtenhofer
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High accuracy optical flow
// estimation based on a theory for warping. ECCV 2004. [2] C. Zach, T. Pock, H.
// Bischof: A duality based approach for realtime TV-L 1 optical flow. DAGM
// 2007.
//************************************************************************

#define N_CHAR 500
#define WRITEOUT_IMGS 1

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>

#include <QDirIterator>
#include <QFileInfo>
#include <QString>

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

// Global variables for gpu::BroxOpticalFlow
const float alpha_ = 0.197;
const float gamma_ = 50;
const float scale_factor_ = 0.8;
const int inner_iterations_ = 10;
const int outer_iterations_ = 77;
const int solver_iterations_ = 10;

static void convertFlowToImage(const Mat &flowIn, Mat &flowOut,
                               float lowerBound, float higherBound) {
#define CAST(v, L, H)                                                          \
  ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255 * ((v) - (L)) / ((H) - (L))))
  for (int i = 0; i < flowIn.rows; ++i) {
    for (int j = 0; j < flowIn.cols; ++j) {
      float x = flowIn.at<float>(i, j);
      flowOut.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
    }
  }
#undef CAST
}

int main(int argc, char *argv[]) {
  GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
  Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1, rgb_out;
  Mat frame0_32, frame1_32, imgU, imgV;
  Mat motion_flow, flow_rgb;

  char cad[N_CHAR];
  struct timeval tod1;
  double t1 = 0.0, t2 = 0.0, tdflow = 0.0, t1fr = 0.0, t2fr = 0.0,
         tdframe = 0.0;


  const char *keys = "{ h  | help            | false | Print help message }"
                     "{ i  | input-dir       | in    | Input directory}"
                     "{ o  | output-dir      | out   | Output directory}"
                     "{ v  | start-video     |  1    | Start video ID }"
                     "{ g  | gpu-id          |  0    | Specify GPU ID to use for computing flow }"
                     "{ f  | method          |  1    | Specify flow method (Brox=0, TVL1=1) }"
                     "{ s  | step            |  1    | Time step between frames for flow calculation}"
                     "{    | min-size        | 256   | Minimum size of the smallest axis of the frame}"
                     "{    | output-size     | 256   | Output size of the smallest axis of the frame}"
                     "{ r  | resize          | true  | Resize frames and flow }"
                     "{    | clip-flow       | true  | Clip flow to interval [-20 20]}"
  ;

  CommandLineParser cmd(argc, argv, keys);

  if (cmd.get<bool>("help")) {
    std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
    std::cerr << "Available options:" << std::endl;
    cmd.printParams();
    return 0;
  }

  std::string vid_path = cmd.get<string>("input-dir");
  std::string out_path = cmd.get<string>("output-dir");
  int start_with_vid = cmd.get<int>("start-video");
  int gpu_id = cmd.get<int>("gpu-id");
  int method = cmd.get<int>("method");
  int frame_step = cmd.get<int>("step");
  float minimum_size = cmd.get<int>("min-size");
  float output_size = cmd.get<int>("output-size");
  bool resize_img = cmd.get<bool>("resize");
  bool clip_flow = cmd.get<bool>("clip-flow"); // clips flow to [-20 20]


  if (out_path.at(out_path.length() - 1) != '/')
    out_path = out_path + "/";

  if (vid_path.at(vid_path.length() - 1) == '/')
    vid_path = vid_path.substr(0, vid_path.length() - 1);

  std::string out_path_jpeg = out_path + "jpegs";

  std::cerr << "Start with video:" << start_with_vid << std::endl
            << "GPU ID:" << gpu_id << std::endl
            << "Flow method: " << (method == 0 ? "Brox" : "TVL1") << std::endl
            << "Number of frames in window for OF: " << frame_step << std::endl
            << "Input folder: " << vid_path << std::endl
            << "Optical flow folder: " << out_path << std::endl
            << "Frames folder: " << out_path_jpeg << std::endl;
  cv::gpu::setDevice(gpu_id);

  cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

  cv::gpu::BroxOpticalFlow dflow(alpha_, gamma_, scale_factor_,
                                 inner_iterations_, outer_iterations_,
                                 solver_iterations_);

  cv::gpu::OpticalFlowDual_TVL1_GPU alg_tvl1;

  QString vpath = QString::fromStdString(vid_path);
  QStringList filters;

  QDirIterator dirIt(vpath, QDirIterator::Subdirectories);

  int vidID = 0;
  std::string video, outfile_u, outfile_v, outfile_jpeg;

  for (; (dirIt.hasNext());) {
    dirIt.next();
    QString file = dirIt.fileName();
    if (file.endsWith("mp4", Qt::CaseInsensitive) || file.endsWith("avi", Qt::CaseInsensitive)) {
      video = dirIt.filePath().toStdString();
    }

    else
      continue;

    vidID++;

    if (vidID < start_with_vid)
      continue;

    std::string fName(video);
    std::string path(video);
    fName.erase(0, vid_path.length());
    path.erase(vid_path.length(), path.length());

    // Remove extension if present.
    const size_t period_idx = fName.rfind('.');
    if (std::string::npos != period_idx)
      fName.erase(period_idx);

    QString out_folder_u = QString::fromStdString(out_path + "u/" + fName);

    bool folder_exists = QDir(out_folder_u).exists();

    if (folder_exists) {
      std::cerr << "already exists: " << out_path << fName << std::endl;
      continue;
    }

    bool folder_created = QDir().mkpath(out_folder_u);
    if (!folder_created) {
      std::cerr << "Cannot create: " << out_path << fName << std::endl;
      continue;
    }

    QString out_folder_v = QString::fromStdString(out_path + "v/" + fName);
    QDir().mkpath(out_folder_v);

    QString out_folder_jpeg = QString::fromStdString(out_path_jpeg + fName);
    QDir().mkpath(out_folder_jpeg);

    std::string outfile = out_path + "u/" + fName + ".bin";

    FILE *fx = fopen(outfile.c_str(), "wb");

    VideoCapture cap;
    try {
      cap.open(video);
    } catch (std::exception &e) {
      std::cerr << e.what() << '\n';
    }

    int nframes = 0, width = 0, height = 0, width_out = 0, height_out = 0;
    float factor = 0, factor_out = 0;

    if (cap.isOpened() == 0) {
      return -1;
    }

    cap >> frame1_rgb_;

    if (resize_img == true) {
      factor =
          std::max<float>(minimum_size / frame1_rgb_.cols, minimum_size / frame1_rgb_.rows);

      width = std::floor(frame1_rgb_.cols * factor);
      width -= width % 2;
      height = std::floor(frame1_rgb_.rows * factor);
      height -= height % 2;

      frame1_rgb = cv::Mat(Size(width, height), CV_8UC3);
      width = frame1_rgb.cols;
      height = frame1_rgb.rows;
      cv::resize(frame1_rgb_, frame1_rgb, cv::Size(width, height), 0, 0,
                 INTER_CUBIC);

      factor_out = std::max<float>(output_size / width, output_size / height);

      rgb_out = cv::Mat(
          Size(cvRound(width * factor_out), cvRound(height * factor_out)),
          CV_8UC3);
      width_out = rgb_out.cols;
      height_out = rgb_out.rows;
    } else {
      frame1_rgb = cv::Mat(Size(frame1_rgb_.cols, frame1_rgb_.rows), CV_8UC3);
      width = frame1_rgb.cols;
      height = frame1_rgb.rows;
      frame1_rgb_.copyTo(frame1_rgb);
    }

    // Allocate memory for the images
    frame0_rgb = cv::Mat(Size(width, height), CV_8UC3);
    flow_rgb = cv::Mat(Size(width, height), CV_8UC3);
    motion_flow = cv::Mat(Size(width, height), CV_8UC3);
    frame0 = cv::Mat(Size(width, height), CV_8UC1);
    frame1 = cv::Mat(Size(width, height), CV_8UC1);
    frame0_32 = cv::Mat(Size(width, height), CV_32FC1);
    frame1_32 = cv::Mat(Size(width, height), CV_32FC1);

    // Convert the image to grey and float
    cvtColor(frame1_rgb, frame1, CV_BGR2GRAY);
    frame1.convertTo(frame1_32, CV_32FC1, 1.0 / 255.0, 0);

    outfile_u = out_folder_u.toStdString();
    outfile_v = out_folder_v.toStdString();
    outfile_jpeg = out_folder_jpeg.toStdString();

    while (frame1.empty() == false) {
      gettimeofday(&tod1, NULL);
      t1fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
      if (nframes >= 1) {
        gettimeofday(&tod1, NULL);
        //	GetSystemTime(&tod1);
        t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
        switch (method) {
        case 0:
          frame1GPU.upload(frame1_32);
          frame0GPU.upload(frame0_32);
          dflow(frame0GPU, frame1GPU, uGPU, vGPU);
        case 1:
          frame1GPU.upload(frame1);
          frame0GPU.upload(frame0);
          alg_tvl1(frame0GPU, frame1GPU, uGPU, vGPU);
        }

        uGPU.download(imgU);
        vGPU.download(imgV);

        gettimeofday(&tod1, NULL);
        t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
        tdflow = 1000.0 * (t2 - t1);
      }

      if (WRITEOUT_IMGS == true && nframes >= 1) {
        if (resize_img == true) {

          cv::resize(imgU, imgU, cv::Size(width_out, height_out), 0, 0,
                     INTER_CUBIC);
          cv::resize(imgV, imgV, cv::Size(width_out, height_out), 0, 0,
                     INTER_CUBIC);
        }

        double min_u, max_u;
        cv::minMaxLoc(imgU, &min_u, &max_u);
        double min_v, max_v;
        cv::minMaxLoc(imgV, &min_v, &max_v);

        float min_u_f = min_u;
        float max_u_f = max_u;

        float min_v_f = min_v;
        float max_v_f = max_v;

        if (clip_flow) {
          min_u_f = -20;
          max_u_f = 20;

          min_v_f = -20;
          max_v_f = 20;
        }

        cv::Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
        cv::Mat img_v(imgV.rows, imgV.cols, CV_8UC1);

        convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
        convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

        sprintf(cad, "/frame%06d.jpg", nframes);

        imwrite(outfile_u + cad, img_u);
        imwrite(outfile_v + cad, img_v);

        fwrite(&min_u_f, sizeof(float), 1, fx);
        fwrite(&max_u_f, sizeof(float), 1, fx);
        fwrite(&min_v_f, sizeof(float), 1, fx);
        fwrite(&max_v_f, sizeof(float), 1, fx);
      }

      sprintf(cad, "/frame%06d.jpg", nframes + 1);
      if (resize_img == true) {
        cv::resize(frame1_rgb, rgb_out, cv::Size(width_out, height_out), 0, 0,
                   INTER_CUBIC);
        imwrite(outfile_jpeg + cad, rgb_out);
      } else
        imwrite(outfile_jpeg + cad, frame1_rgb);

      std::cerr << "Writing: " << outfile_jpeg+cad << std::endl;

      frame1_rgb.copyTo(frame0_rgb);
      cvtColor(frame0_rgb, frame0, CV_BGR2GRAY);
      frame0.convertTo(frame0_32, CV_32FC1, 1.0 / 255.0, 0);

      nframes++;
      for (int iskip = 0; iskip < frame_step; iskip++) {
        cap >> frame1_rgb_;
      }
      if (frame1_rgb_.empty() == false) {
        if (resize_img == true) {
          cv::resize(frame1_rgb_, frame1_rgb, cv::Size(width, height), 0, 0,
                     INTER_CUBIC);
        } else {
          frame1_rgb_.copyTo(frame1_rgb);
        }

        cvtColor(frame1_rgb, frame1, CV_BGR2GRAY);
        frame1.convertTo(frame1_32, CV_32FC1, 1.0 / 255.0, 0);
      } else {
        break;
      }

      gettimeofday(&tod1, NULL);
      t2fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
      tdframe = 1000.0 * (t2fr - t1fr);
      std::cerr << "Processing video: " << fName << std::endl
                << "   ID: "<< vidID << std::endl
                << "   Frame number: " << nframes << std::endl
                << "   Flow method: " << method <<  std::endl
                << "   Time computing OF: " << tdflow << " ms" << std::endl
                << "   Time All: " << tdframe << " ms" << std::endl;
    }
    fclose(fx);
  }

  return 0;
}
