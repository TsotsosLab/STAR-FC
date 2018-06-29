
/*
 *  * Author: yulia
 * Date: 18/11/2016
 */
#ifndef CONSPICUITY_MAP_HPP_
#define CONSPICUITY_MAP_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

using namespace cv;
using namespace std;

class ConspicuityMap {
private:
	int height, width;
	Mat conspMap;
	double periphGain, centerGain;
	double scaledPeriphMapMax, scaledCentralMapMax;
public:
	ConspicuityMap(int h, int w, double pgain, double cgain);
	~ConspicuityMap();

	void reset(int h, int w);

	void computeConspicuityMap(Mat periphMap, Mat centralMap);

	Mat getConspMap();
	void setPeripheralGain(double pgain);
	void setCenterGain(double cgain);
	double getScaledPeriphMapMax();
	double getScaledCentralMapMax();
};

#endif
