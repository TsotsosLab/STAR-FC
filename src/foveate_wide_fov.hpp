/*
Author: yulia
Date: 1/04/2016
*/

#ifndef FOVEATE_WIDE_H
#define FOVEATE_WIDE_H

#include <cstdio>
#include <string>
#include <math.h>
#include <vector>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "utils.hpp"

#ifdef DEBUG
#include "matio.h"
#endif


#include "ba_interp3.hpp"

#ifdef GPU_FOVEATION
#include "ba_interp3.cuh"
typedef float data_t;
#define CV_DATA_T CV_32F
#else
typedef double data_t;
#define CV_DATA_T CV_64F
#endif

using namespace std;
using namespace cv;

//Calden: this is a modification of Foveate for 160 field of view
//things that have changed are parameters of the cones and rods functions
//and dotpitch

class Foveate_wide_fov {
    public:
      //input image should be double grayscale or color
      //gaze_pos is a 2D coordinate within the image
      //rods_and_cones - if set to true will use both rods and cones models
      // if set to false, only the cones model will be used
	  //dot pitch - physical size of one pixel in meters
	  //viewingdist - distance from the screen in meters
     //double dotpitch = 0.0063; //dot pitch for image 1024 px wide to cover 160 deg visual field
     //double viewingdist = 0.57; 			// viewing distance in meters
	Foveate_wide_fov();
	~Foveate_wide_fov();
	Mat foveate(Mat input_img, Point gaze_pos, double dotpitch, double viewingdist, bool rods_and_cones);
private:
	//number of levels in visual pyramid
	int num_levels;

	//bool use_ycbcr = true;
	double CTO;    // constant from Geisler&Perry
	double alpha; 			// constant from Geisler&Perry
	double epsilon2; 			  // constant from Geisler&Perry

	Mat input_img, output_img;
	int orig_h, orig_w;
	Mat xi, yi;
	double *xi_buf, *yi_buf;
	data_t *fov_cones_buf, *fov_rods_buf;
	Mat pyrlevel_cones, pyrlevel_rods;
	data_t *pyrlevel_cones_buf, *pyrlevel_rods_buf;
	vector<Mat> pyramid;
	data_t *pyramid_buf;
	// ex and ey are the x- and y- offsets of each pixel compared to
	// the point of focus (fovx,fovy) in pixels.
	Mat ex, ey, eradius, dist_px, ec;
	double min_val, max_val;
	Point p_min, p_max;
	Mat eyefreq_cones, eyefreq_rods;
	double p[6] = {8.8814e-11,  -1.6852e-07,   1.1048e-04,  -3.1856e-02,   3.7501e+00,  -3.0283e+00};
	Mat channels[3];
	Mat output[3];

	void init();
	void setImage(Mat input_img);
	void preprocess(Mat input_img, Point gaze_pos, double dotpitch, double viewingdist, bool rods_and_cones);
	void cleanup();
	void compute_image_pyramid_alloc(Mat img);
	void compute_meshgrid();
	Mat ggamma(Mat X, double alpha, double beta, double g, double sigma, double mu);
	Mat polyval(Mat x, double *p, int nc);
};
#endif
