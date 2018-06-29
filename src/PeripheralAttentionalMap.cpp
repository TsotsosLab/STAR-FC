/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "PeripheralAttentionalMap.hpp"


PeripheralAttentionalMap::PeripheralAttentionalMap(int h, int w, double deg2pix, double pSizeDeg, string salAlg, string AIMBasis) {
	//initialize AIM with the basis function
	//TODO: change this to take in mat files instead
	this->deg2pix = deg2pix;
	periphFieldSizeDeg = pSizeDeg;
	periphFieldSizePix = periphFieldSizeDeg/deg2pix;

	if (salAlg.compare("AIM") == 0) {
		algo = AIM_BU;
		aim = new AIM();
		char basis[200];
		sprintf(basis, "data/%s.bin", AIMBasis.c_str());
		aim->loadBasis(basis);

	} else if (salAlg.compare("BMS") == 0){
		algo = BMS_BU;
	} else {
		algo = VOCUS_BU;
		vocus2CFG = new VOCUS2_Cfg();
		vocus2CFG->load("contrib/vocus2-version1.1/cfg/msra_cfg.xml");
		vocus2 = new VOCUS2(*vocus2CFG);
	}
	initPeripheralMask(h, w);
}

PeripheralAttentionalMap::~PeripheralAttentionalMap() {
	if (algo == AIM_BU) {
		delete aim;
	}
}

void PeripheralAttentionalMap::reset(int h, int w) {
	initPeripheralMask(h, w);
	periphMap = Scalar(0);
}

void PeripheralAttentionalMap::initPeripheralMask(int h, int w){
	periphMask = Mat(h, w, CV_8U);

	int centX = h/2;
	int centY = w/2;
	double radius;

	for(int i=0; i < h; i++){
		for(int j=0; j < w; j++){
			radius = sqrt((double)((i-centX)*(i-centX) + (j-centY)*(j-centY)));
			if (radius <= periphFieldSizePix){
				periphMask.at<char>(i,j) = 0;
			} else {
				periphMask.at<char>(i,j) = 1;
			}
		}
	}
}

void PeripheralAttentionalMap::computeBUSaliency(Mat view) {
	chrono::time_point<chrono::system_clock> start, end;
	start = chrono::system_clock::now();
	switch(algo) {
	case AIM_BU:
		computeAIMSaliency(view);
		break;
	case BMS_BU:
		computeBMSSaliency(view);
		break;
	case VOCUS_BU:
		computeVOCUSSaliency(view);
		break;
	default:
		computeAIMSaliency(view);
		cout << "Unrecognized BU Saliency algorithm! Running AIM instead!" << endl;
		break;
	}
	end = chrono::system_clock::now();
	chrono::duration<double> elapsedSec = end-start;
	cout << "Saliency took " << elapsedSec.count() << " sec" << endl;
}

void PeripheralAttentionalMap::computeVOCUSSaliency(Mat view) {
	vocus2->process(view);
	//if(CENTER_BIAS)
	//  salmap = vocus2.add_center_bias(0.00005);
	//else
	VOCUSSalMap = vocus2->get_salmap();
	VOCUSSalMap.convertTo(VOCUSSalMap, CV_64FC1);
	VOCUSSalMap.copyTo(salMap);
}

void PeripheralAttentionalMap::computeBMSSaliency(Mat view) {

	/* Default BMS parameters for fixation prediction
	 * taken from the BMS.m in contrib/BMS_v2 */
	double sample_step     =   8; //delta
	int max_dim              =   400; // maximum dimension of the image

	//do not change the following

	int dilation_width_1     =   max((int)round(7*max_dim/400),1); //omega
	int dilation_width_2     =   max((int)round(9*max_dim/400),1); //kappa
	float blur_std             =   round(9*max_dim/400.0); //sigma
	int color_space          =   2; //RGB: 1; Lab: 2; LUV: 4
	int whitening            =   1; //do color whitening

	/*Note: we transform the kernel width to the equivalent iteration
	number for OpenCV's **dilate** and **erode** functions**/
	int DILATION_WIDTH_1	=	(dilation_width_1-1)/2;//3: omega_d1
	int DILATION_WIDTH_2	=	(dilation_width_2-1)/2;//11: omega_d2

	int NORMALIZE			=	1 /*atoi(argv[7])*/;//1: whether to use L2-normalization
	int HANDLE_BORDER		=	0 /*atoi(argv[8])*/;//0: to handle the images with artificial frames



	Mat view_resized;
	view.copyTo(view_resized);

	//pre- and post-processing steps are the same as in mexBMS.cpp in contrib/BMS

	//pre-processing
	float maxD = max(view.cols, view.rows);
	resize(view_resized, view_resized, Size((int)(max_dim*view.cols/maxD), (int)(max_dim*view.rows/maxD)), 0.0, 0.0, INTER_AREA);
	view.convertTo(view_resized, CV_8U);

	BMS bms(view_resized, DILATION_WIDTH_1, NORMALIZE, HANDLE_BORDER, color_space, whitening);
	bms.computeSaliency((double)sample_step);
	Mat result = bms.getSaliencyMap();

	/* Post-processing */
	if (DILATION_WIDTH_2>0)
		dilate(result,result,Mat(),Point(-1,-1),DILATION_WIDTH_2);
	if (blur_std > 0)
	{
		int blur_width = MIN(floor(blur_std)*4+1, 51);
		GaussianBlur(result,result,Size(blur_width,blur_width),blur_std,blur_std);
	}

	//resize to the original size
	result.convertTo(result, CV_64FC1);
	resize(result, result, view.size(), 0.0, 0.0, INTER_CUBIC);
	normalize(result, result, 0, 1.0, NORM_MINMAX, CV_64F);
	result.copyTo(salMap);
}

void PeripheralAttentionalMap::computeAIMSaliency(Mat view) {
	aim->loadImage(view, 0.5f);
	aim->runAIM();
	aim->adj_sm.copyTo(salMap);
}

void PeripheralAttentionalMap::computePeriphMap(bool mask) {

	Mat blurredPeriphMap;
	GaussianBlur(salMap, blurredPeriphMap, Size(11, 11), 0, 0);
	if (mask) {
		blurredPeriphMap.copyTo(periphMap, periphMask);
	} else {
		blurredPeriphMap.copyTo(periphMap);
	}

	//saveToImg("output/apples/peripheral_map.png", blurredPeriphMap);
	//normalize(periphMap, periphMap, 0, 1, NORM_MINMAX, CV_64F);
}

double PeripheralAttentionalMap::getPeriphFieldSizePix() {
	return periphFieldSizePix;
}

Mat PeripheralAttentionalMap::getPeriphMap() {
	return periphMap;
}

double PeripheralAttentionalMap::getPeriphMapMax() {
	double maxVal;
	minMaxLoc(periphMap, NULL, &maxVal, NULL, NULL);
	return maxVal;
}
