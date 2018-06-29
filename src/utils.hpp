#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "matio.h"
#include <fftw3.h>
#include "factorize.h"



using namespace std;
using namespace cv;
void show_mat(const char* window_name, Mat img, int wait);
void saveToImg(const char* file_name, Mat img);
void saveToMat(const char* file_name, Mat img);
void saveToMat(const char* file_name, vector<Point> points);
//void savePixmap(const char* file_name, QPixmap pixmap);
void printMinMax(const char* img_name, Mat img);
Point swapPointCoords(Point p);
Mat convolve(Mat img, Mat kernel);

#endif
