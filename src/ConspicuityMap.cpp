/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "ConspicuityMap.hpp"

ConspicuityMap::ConspicuityMap(int h, int w, double pgain, double cgain) {
	this->height = h;
	this->width = w;
	periphGain = pgain;
	centerGain = cgain;
}

ConspicuityMap::~ConspicuityMap(){

}

void ConspicuityMap::reset(int h, int w) {
	height = h;
	width = w;
}

void ConspicuityMap::computeConspicuityMap(Mat periphMap, Mat centralMap){
	Mat scaledPeriphMap, scaledCentralMap;

	multiply(periphMap, Scalar(periphGain), scaledPeriphMap);
	multiply(centralMap, Scalar(centerGain), scaledCentralMap);

	minMaxLoc(scaledPeriphMap, NULL, &scaledPeriphMapMax, NULL, NULL);
	minMaxLoc(scaledCentralMap, NULL, &scaledCentralMapMax, NULL, NULL);
//	printMinMax("periphMap", scaledPeriphMap);
//	printMinMax("centralMap", scaledCentralMap);

//  cout << "Combining the maps \n";
//  cout << "Size of PeriphMap: " << scaledPeriphMap.size() << " and channels: " << scaledPeriphMap.channels() << "\n";
//  //cout << "Number of dimensions of PeriphMap: " << scaledPeriphMap.dims << "\n";
//  cout << "Size of CentralMap: " << scaledCentralMap.size() << " and channels: " << scaledCentralMap.channels() << "\n";
//  //cout << "Number of dimensions of CentralMap: " << scaledCentralMap.dims << "\n";

	max(scaledPeriphMap, scaledCentralMap, conspMap);

	//printMinMax("conspMap", conspMap);
	//show_mat("Conspicuity Map", conspMap, 0);
}

Mat ConspicuityMap::getConspMap(){
	return conspMap;
}

double ConspicuityMap::getScaledPeriphMapMax() {
	return scaledPeriphMapMax;
}

double ConspicuityMap::getScaledCentralMapMax() {
	return scaledCentralMapMax;
}

void ConspicuityMap::setPeripheralGain(double pgain) {
	periphGain = pgain;
}

void ConspicuityMap::setCenterGain(double cgain) {
	centerGain = cgain;
}
