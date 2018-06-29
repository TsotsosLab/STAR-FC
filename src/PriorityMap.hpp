/*
 * Author: yulia
 * Date: 18/11/2016
 */
#ifndef PRIORITY_MAP_HPP_
#define PRIORITY_MAP_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "PeripheralAttentionalMap.hpp"
#include "CentralAttentionalMap.hpp"
#include "utils.hpp"
#include <random>

using namespace cv;
using namespace std;

class PriorityMap {
private:
	Mat priorityMap, priorityMapCopy;
	Mat priorityMapVis;
	Mat dist;
	Point nextFixationDirection;
	int width, height;
	double periphGain, centerGain;
	int blendingStrategy;
	bool nextFixationAsMaximum;
	double nextFixationThreshold;

public:

	PriorityMap(int h, int w, double pgain, double cgain, int blendStrategy, bool deterministicNextFix, double thresh);
	~PriorityMap();

	void reset(int h, int w);
	void initDist();

	void combinePeriphAndCentralWeighted(Mat periphMap, Mat centralMap, Mat fixHistMap, double periphFieldSizePix);
	void combinePeriphAndCentralMax(Mat periphMap, Mat centerMap, Mat fixHistMap);
	void computeNextFixationDirection(PeripheralAttentionalMap* pMap, CentralAttentionalMap* cMap, Mat fixHistMap);
	Point getNextFixationDirection();
	Mat getPriorityMap();
	Mat getPriorityMapVis();

	void setPeripheralGain(double pgain);
	void setCenterGain(double cgain);

	double getPeripheralGain();
	double getCenterGain();
};


#endif
