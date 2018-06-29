/*
 * Author: yulia
 * Date: 18/11/2016
 */
#ifndef FIXATION_HISTORY_MAP_HPP_
#define FIXATION_HISTORY_MAP_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

using namespace cv;
using namespace std;


class FixationHistoryMap {
private:
	int height, width, heightFull, widthFull;
	Mat fixHistMap;
	Mat fixHistMapFull; //fixations over the whole environment

	double iorSizeDeg;
	double iorSizePx;
	double decayRate;

	vector<Point> fixationsList;
	Point lastFixation;

public:

	FixationHistoryMap(int h, int w, int envH, int envW, double iorDecayRate, double iorSizeDeg, double deg2pix);
	~FixationHistoryMap();
	void reset(int h, int w, int envH, int envW);
	void saveFixationCoords(Point fixCoords);
	Mat getFixationHistoryMap();
	Mat getFixationHistoryMap(Point gaze);
	void decayFixations();
	int getNumberOfFixations();
	void dumpFixationsToMat(const char* filename);
};

#endif
