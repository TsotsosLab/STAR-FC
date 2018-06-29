/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "FixationHistoryMap.hpp"

FixationHistoryMap::FixationHistoryMap(int h, int w, int envH, int envW, double iorDecayRate, double iorSizeDeg, double deg2pix) {
	height = h;
	width = w;
	heightFull = envH;
	widthFull = envW;

	this->iorSizeDeg = iorSizeDeg;
	this->decayRate = iorDecayRate;
	this->iorSizePx = iorSizeDeg/deg2pix;

	fixHistMap = Mat(h, w, CV_64F, Scalar(0));
	fixHistMapFull = Mat(envH, envW, CV_64F, Scalar(0));
	lastFixation = Point(width/2, height/2);
}


FixationHistoryMap::~FixationHistoryMap() {

}

void FixationHistoryMap::reset(int h, int w, int envH, int envW) {
	height = h;
	width = w;
	heightFull = envH;
	widthFull = envW;

	fixHistMap = Mat(h, w, CV_64F, Scalar(0));
	fixHistMapFull = Mat(envH, envW, CV_64F, Scalar(0));
	lastFixation = Point(width/2, height/2);
	fixationsList.clear();
}

void FixationHistoryMap::saveFixationCoords(Point fixCoords) {
	//add fixCoords to the list of fixations
	lastFixation = fixCoords;
	Point fixCoordsFull = fixCoords + Point(width/2, height/2);
	fixationsList.push_back(fixCoords);

	//cout << "FixCoords x=" << fixCoords.y << " y=" << fixCoords.x << endl;
	//cout << "FixCoordsFull x=" << fixCoordsFull.y << " y=" << fixCoordsFull.x << endl;
	//add inhibition area around the new fixation with radius iorSizePx
	for(int i = fixCoordsFull.y-iorSizePx; i < fixCoordsFull.y+iorSizePx; i++){
		for(int j = fixCoordsFull.x-iorSizePx; j < fixCoordsFull.x+iorSizePx; j++){
			double d = pow(1.0*(fixCoordsFull.y-i),2) + pow(1.0*(fixCoordsFull.x-j),2);
			d = sqrt(d);
			if(d <= iorSizePx){
				fixHistMapFull.ptr<double>(i)[j] = min(1.0, fixHistMapFull.ptr<double>(i)[j] + 1 - d/iorSizePx); //clamp to 1.0
			}
		}
	}
//	saveToImg("debug/fixHistMapFull.png", fixHistMapFull);
//	saveToImg("debug/fixHistMapTest.png", fixHistMapFull(Rect(width/2, height/2, width, height)));
}

//add decay to all existing fixations
//clamp negative values to 0
void FixationHistoryMap::decayFixations() {
	Mat temp;
	subtract(fixHistMapFull, Scalar(1.0/decayRate), temp);
	max(temp, Scalar(0), fixHistMapFull);
}

Mat FixationHistoryMap::getFixationHistoryMap() {
	//this should return a portion of the fixation history map
	//based on the current gaze coordinate
	//cout << "fixhistmap prev_fixation x=" << lastFixation.y << " y=" << lastFixation.x << endl;
	return fixHistMapFull(Rect(lastFixation.x, lastFixation.y, width, height));
}

Mat FixationHistoryMap::getFixationHistoryMap(Point gaze) {
	return fixHistMapFull(Rect(gaze.x, gaze.y, width, height));
}

int FixationHistoryMap::getNumberOfFixations() {
	return fixationsList.size();
}

void FixationHistoryMap::dumpFixationsToMat(const char* filename) {
	saveToMat(filename, fixationsList);
}

