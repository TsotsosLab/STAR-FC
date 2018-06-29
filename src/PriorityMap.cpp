/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "PriorityMap.hpp"


PriorityMap::PriorityMap(int h, int w, double pgain, double cgain, int blendStrategy, bool deterministicNextFix, double thresh) {
	priorityMap = Mat(h, w, CV_64F, Scalar(0));
	dist = Mat(h, w, CV_64F, Scalar(0));
	width = w;
	height = h;
	periphGain = pgain;
	centerGain = cgain;
	nextFixationDirection = Point(-1, -1);
	blendingStrategy = blendStrategy;
	nextFixationAsMaximum = deterministicNextFix; //how next fixation is determined
	nextFixationThreshold = thresh;
	initDist();
}

void PriorityMap::reset(int h, int w) {
	priorityMap = Mat(h, w, CV_64F, Scalar(0));
	dist = Mat(h, w, CV_64F, Scalar(0));
	width = w;
	height = h;
	nextFixationDirection = Point(-1, -1);
	initDist();
}


void PriorityMap::initDist() {
	int centX = height/2;
	int centY = width/2;

	for(int i=0; i < height; i++){
		for(int j=0; j < width; j++){
			dist.at<double>(i, j) = sqrt((double)((i-centX)*(i-centX) + (j-centY)*(j-centY)));
		}
	}
}

void PriorityMap::combinePeriphAndCentralWeighted(Mat periphMap, Mat centralMap, Mat fixHistMap, double periphFieldSizePix) {

	double r_max, r_min;
	double d_gain = 1.1; // A decision gain to try to prevent AIM from overriding SALICON's decisions in the centre
	minMaxLoc(dist, NULL, &r_max);
	double r_c = periphFieldSizePix;
	for (int i=0; i < height; i++) {
		for (int j=0; j<width; j++) {
			double r_p = dist.at<double>(i, j);
			if (r_p < r_c) {
//				if (periphMap.at<double>(i,j) > (d_gain*(r_c-r_p)/r_c)*centralMap.at<double>(i,j) ){
//					priorityMap.at<double>(i, j) = periphMap.at<double>(i, j);
//				}else{
//					priorityMap.at<double>(i, j) = centralMap.at<double>(i,j);
//				}
				priorityMap.at<double>(i, j) = ((r_c-r_p)/r_c)*centralMap.at<double>(i,j) + (r_p/r_c)*periphMap.at<double>(i, j);
			} else {
				priorityMap.at<double>(i, j) = (1 + ((r_p-r_c)/(r_max-r_c))*(abs(1-periphGain)))*periphMap.at<double>(i, j);
			}
		}
	}

	subtract(priorityMap, fixHistMap, priorityMap);
}


void PriorityMap::combinePeriphAndCentralMax(Mat periphMap, Mat centralMap, Mat fixHistMap) {
	Mat scaledPeriphMap, scaledCentralMap;
	multiply(periphMap, Scalar(periphGain), scaledPeriphMap);
	multiply(centralMap, Scalar(centerGain), scaledCentralMap);

	Mat periphMapInhibit, centralMapInhibit;
	//subtract fixation history map from peripheral and center maps
	//to inhibit them and take element-wise max of the two

	subtract(scaledPeriphMap, fixHistMap, periphMapInhibit);
	subtract(scaledCentralMap, fixHistMap, centralMapInhibit);
	max(periphMapInhibit, centralMapInhibit, priorityMap);
}

void PriorityMap::computeNextFixationDirection(PeripheralAttentionalMap* pMap, CentralAttentionalMap* cMap, Mat fixHistMap) {
	if (blendingStrategy == 3) {
		combinePeriphAndCentralWeighted(pMap->getPeriphMap(), cMap->getCentralMap(), fixHistMap, pMap->getPeriphFieldSizePix());
	} else {
		combinePeriphAndCentralMax(pMap->getPeriphMap(), cMap->getCentralMap(), fixHistMap);
	}
	double minVal, maxVal;
	minMaxLoc(priorityMap, &minVal, &maxVal, NULL, NULL);
	printf("Priority Map: [%.03f %.03f]\n", minVal, maxVal);


	if (nextFixationAsMaximum) {
		//find location of the maximum
		//compute direction of the next fixation relative to the
		//center of the image (current fixation)
		Point maxLoc;
		minMaxLoc(priorityMap, NULL, NULL, NULL, &maxLoc);

		nextFixationDirection = maxLoc - Point(width/2, height/2);
		//cout << "nextFixDirection (" << nextFixationDirection.y << ", " << nextFixationDirection.x << ")" << endl;
	} else {
		double globalMax;
		Mat nonZeroLoc;
		Mat priorityMapBinary;
		priorityMap.copyTo(priorityMapCopy);
		minMaxLoc(priorityMapCopy, NULL, &globalMax, NULL, NULL);
		threshold(priorityMapCopy, priorityMapBinary, globalMax*nextFixationThreshold, 1, THRESH_BINARY);
		priorityMapBinary.convertTo(priorityMapBinary, CV_8UC1);
		findNonZero(priorityMapBinary, nonZeroLoc);

		//show_mat("priorityMapBinary",priorityMapBinary, 5);

		random_device rd;
		mt19937 rng(rd());
		uniform_int_distribution<int> uni(0, nonZeroLoc.total());
		int randLocIdx = uni(rng);

		nextFixationDirection = nonZeroLoc.at<Point>(randLocIdx) - Point(width/2, height/2);
		//cout << "nextFixDirection (" << nextFixationDirection.y << ", " << nextFixationDirection.x << ")" << endl;
	}
//	saveToImg("debug/priorityMap.png", priorityMap);
//	saveToMat("debug/priorityMap.mat", priorityMap);
}


Point PriorityMap::getNextFixationDirection() {
	return nextFixationDirection;
}

Mat PriorityMap::getPriorityMap() {
	return priorityMap;
}

Mat PriorityMap::getPriorityMapVis() {
	//for visualization raise the map to the power of 4
	//to make maxima more visible
	pow(priorityMap, 2, priorityMapVis);
	return priorityMapVis;
}

void PriorityMap::setCenterGain(double cgain) {
	centerGain = cgain;
}

void PriorityMap::setPeripheralGain(double pgain) {
	periphGain = pgain;
}


double PriorityMap::getCenterGain() {
	return centerGain;
}

double PriorityMap::getPeripheralGain() {
	return periphGain;
}
