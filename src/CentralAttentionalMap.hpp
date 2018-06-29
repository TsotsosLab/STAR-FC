#ifndef CENTRAL_ATT_MAP_HPP_
#define CENTRAL_ATT_MAP_HPP_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <matio.h>
#include <limits>
#include "factorize.h"
#include "utils.hpp"


#ifdef WITH_SALICON
#include "Salicon.hpp"
#endif

using namespace cv;
using namespace std;

class CentralAttentionalMap {
private:
	int height, width;

	Mat centralMask;
	Mat centralMap;

	vector<Mat> faceTemplates;
	vector<Mat> faceDetections;
	Mat faceDetectionResult;
	Mat centralSaliencyMap;

	double deg2pix;
	double centralFieldSizeDeg;
	double centralFieldSizePix;

#ifdef WITH_SALICON
	Salicon* salicon;
#endif

	//static QMutex fftMutex;
public:

	CentralAttentionalMap(int h, int w, double deg2pix, double cSizeDeg);
	~CentralAttentionalMap();

	void reset(int h, int w);

	void centralDetection(Mat view);

	void loadFaceTemplates();
	void initCentralMask(int h, int w);
	Mat getCentralMap();
	double getCentralMapMax();
	void detectFaces(Mat view);
	void maskFaceDetection();
	void maskCentralDetection();
};

#endif //CENTRAL_ATT_MAP_HPP_
