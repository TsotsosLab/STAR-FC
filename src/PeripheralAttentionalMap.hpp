/*
 * Author: yulia
 * Date: 18/11/2016
 */
#ifndef PERIPH_ATT_MAP_HPP_
#define PERIPH_ATT_MAP_HPP_


#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include "BMS.h"
#include "utils.hpp"
#include "AIM.h"
#include "VOCUS2.h"

using namespace cv;
using namespace std;

enum BUSalAlgo{ AIM_BU = 0, BMS_BU = 1, VOCUS_BU = 2};

class PeripheralAttentionalMap {
private:

	AIM* aim;
	Mat AIMSalMap;

	BMS* bms;
	Mat BMSSalMap;


	VOCUS2_Cfg* vocus2CFG; //configuration class for VOCUS2
	VOCUS2* vocus2;
	Mat VOCUSSalMap;

	Mat salMap;

	Mat periphMask;
	Mat periphMap;

	BUSalAlgo algo;

	float deg2pix;
	float periphFieldSizeDeg;
	float periphFieldSizePix;

public:

	PeripheralAttentionalMap(int h, int w, double deg2pix, double pSizeDeg, string salAlg, string AIMBasis);
	~PeripheralAttentionalMap();
	void reset(int h, int w);
	void initPeripheralMask(int h, int w);
	void computeBUSaliency(Mat view);
	void computeAIMSaliency(Mat view);
	void computeBMSSaliency(Mat view);
	void computeVOCUSSaliency(Mat view);
	void maskBUSaliency();
	void computePeriphMap(bool mask);
	Mat getPeriphMap();
	double getPeriphMapMax();
	double getPeriphFieldSizePix();

};

#endif
