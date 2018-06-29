/*
 * AIM.h
 *
 *  Created on: 2015-07-10
 *      Author: yulia
 */

#ifndef AIM_H_
#define AIM_H_

//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#include <string.h>
#include <sys/stat.h>
#include <stdio.h>
#include <matio.h>
#include <chrono>

#include "utils.hpp"

using namespace cv;
using namespace std;

class AIM {
public:
	AIM();
	virtual ~AIM();

	void loadBasis(char* filename);
	void loadBasisMat(const char* filename);
	void loadImage(const char* filename, float scale_factor);
	void loadImage(Mat image, float scale_factor);
	void update();
	void runAIM();
	void saveResult(const char* filename);
	Mat adj_sm;
private:
	Mat **kernels;

	Ptr<cuda::Convolution> convolver;

	float* kernel_data;
	float scale;
	vector<Mat> channels;
	Mat image, temp, temp1, sm, hist;
	vector<Mat> aim_temp;
	int num_kernels, kernel_size, num_channels;

	double max, min, max_aim, min_aim;
	int orig_h, orig_w; //original dimensions of the image
	int new_h, new_w;
};


#endif /* AIM_H_ */
