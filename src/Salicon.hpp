#ifndef SALICON_HPP
#define SALICON_HPP

#include <stdio.h>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace caffe;
using namespace std;
using namespace cv;

class Salicon {

public:
	Salicon();
	~Salicon();
	void initNet();
	void loadImage(string imgName);
	void loadImage(Mat img);
	void processImage();
	void cropImageToAspectRatio();
	void initMeanImg();
	Mat computeSalMap();

private:
	Net<float>* net;
	Mat img, croppedImg, coarseImg, fineImg, meanImg, salMap;

	vector<Mat> fineInputChannels, coarseInputChannels;

	string protoFilename = "contrib/OpenSALICON/salicon.prototxt";
	string caffemodelFilename = "contrib/OpenSALICON/salicon_osie.caffemodel";

	int salMapBlobIdx, coarseInputBlobIdx, fineInputBlobIdx;
	float* salMapData;
};

#endif
