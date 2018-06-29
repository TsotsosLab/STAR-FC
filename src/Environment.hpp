/*
 * Author: yulia
 * Date: 18/11/2016
 */
#ifndef ENVIRONMENT_HPP_
#define ENVIRONMENT_HPP_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.hpp"

#define PI 3.14159265

using namespace cv;
using namespace std;

enum DOT_PITCH {FROM_DEG2PIX, FROM_VIS_ANGLE_SIZE};

/* This class loads and displays external stimuli. For now static only */

class Environment {

private:
	Mat scene;
	Mat scenePadded;
	Mat sceneWithFixations;

	Point prevFix;

	/*Parameters of the input image */
	int width, height, widthPadded, heightPadded; //dimensions of the image in pixels
	double widthm; //width of the stimuli in meters

	/*Viewing parameters */
	double viewingDist; //distance from the stimuli in meters
	double dotPitch; //physical size of the pixel in meters
	double inputSizeDeg; //size of the stimuli in deg vis angle
	double pix2deg; //how many pixels per deg vis angle
	Scalar paddingRGB; //color for padding, if set to -1 the average of the image is used instead
	DOT_PITCH dotPitchMethod; //method to calculate dotpitch

public:
	Environment();
	Environment(double viewingDist, int screenWidthpx, int screenHeightpx, int diagIn);
	Environment(double viewingDist, double inputSizeDeg, double deg2pix, Scalar paddingRGB);
	~Environment();
	void loadStaticStimulus(string filename);
	void padStaticStimulus();
	Mat getPaddedScene();
	Mat getScene();
	Mat getSceneWithFixations();
	void saveSceneWithFixations(const char* fileName);
	Mat getEyeView(Point currentGazeCoords, int width, int height);
	double getViewingDist();
	double getInputSizeDeg();
	double getPix2Deg();
	double getDotPitch();
	void updateDotPitch();
	void drawFixation(Point fix);

	int getHeight();
	int getWidth();
	int getFullHeight();
	int getFullWidth();
};

#endif /* ENVIRONMENT_HPP_ */
