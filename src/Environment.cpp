/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "Environment.hpp"

Environment::Environment(){
	//if no viewing distance is provided, assume 57 cm

	viewingDist = 1.06; //default viewing distance (taken from CAT2000 setup)
	inputSizeDeg = 45; // default size of the stimulus in deg vis angle (taken from CAT2000 setup)
	widthm = 2*viewingDist*tan((inputSizeDeg*PI/180.0)/2);
	pix2deg = -1;
	dotPitchMethod = FROM_VIS_ANGLE_SIZE;
}

//provide monitor parameters and viewing distance
//diagonal in inches
Environment::Environment(double viewingDist, int screenWidthpx, int screenHeightpx, int diagIn){
	double d = sqrt(screenHeightpx*screenHeightpx + screenWidthpx*screenWidthpx);
	double PPI = d/diagIn;
	dotPitch = 0.0254/PPI;
}

//provide viewing distance and either degrees of visual angle that the stimuli should span or
//desired number of pixels per dva (specify one argument and set another one to -1)
Environment::Environment(double viewingDist, double inputSizeDeg, double pix2deg, Scalar paddingRGB) {
	if (inputSizeDeg) {
		this->viewingDist = viewingDist;
		this->inputSizeDeg = inputSizeDeg;
		widthm = 2*viewingDist*tan((inputSizeDeg*PI/180.0)/2); //calculate width of the image in meters
		dotPitchMethod = FROM_VIS_ANGLE_SIZE;
	} else {
		this->viewingDist = viewingDist;
		this->pix2deg = pix2deg;
		dotPitchMethod = FROM_DEG2PIX;
	}
	this->paddingRGB = paddingRGB;

}

Environment::~Environment() {

}

void Environment::loadStaticStimulus(string filename) {

	scene = imread(filename, CV_LOAD_IMAGE_COLOR);
	if (!scene.data) {
		cout << "file " << filename << " could not be opened!" << endl;
	}
	scene.copyTo(sceneWithFixations);

	Size size = scene.size();
	width = size.width;
	height = size.height;
	prevFix = Point(width/2, height/2);
	padStaticStimulus();

	updateDotPitch();

//	namedWindow("Environment", WINDOW_NORMAL);
//	imshow("Environment", scene_padded);
//	waitKey(0);
}


void Environment::padStaticStimulus() {
	Scalar avgPix;
	//find average color of the image
	if (paddingRGB[0] < 0) {
		avgPix = mean(scene);
	} else {
		avgPix = paddingRGB;
	}

	int top, bottom, left, right;
	top = bottom = height/2;
	left = right = width/2;
	copyMakeBorder(scene, scenePadded, top, bottom, left, right, BORDER_CONSTANT, avgPix);
	heightPadded = scenePadded.rows;
	widthPadded = scenePadded.cols;
	//cout << scenePadded.size() << endl;
}

//returns a full environment
Mat Environment::getPaddedScene() {
	return scenePadded;
}

Mat Environment::getScene() {
	return scene;
}

void Environment::drawFixation(Point newFix) {

	//line(sceneWithFixations, prevFix+Point(1, 0), newFix+Point(1, 0), CV_RGB(50, 50, 50), 3, CV_AA, 0);
	line(sceneWithFixations, prevFix, newFix, CV_RGB(250, 62, 62), 2, CV_AA, 0);
	circle(sceneWithFixations, prevFix, 3, CV_RGB(250, 62, 62), -1, CV_AA, 0);
	circle(sceneWithFixations, newFix, 3, CV_RGB(250, 62, 62), -1, CV_AA, 0);
	prevFix = newFix;
}

void Environment::saveSceneWithFixations(const char* fileName) {
	saveToImg(fileName, sceneWithFixations);
}

Mat Environment::getSceneWithFixations() {
	return sceneWithFixations;
}
//return what the eye sees
Mat Environment::getEyeView(Point currentGazeCoords, int width, int height) {
	return scenePadded(Rect(currentGazeCoords.x, currentGazeCoords.y, width, height));
}


int Environment::getHeight() {
	return height;
}

int Environment::getWidth() {
	return width;
}

double Environment::getDotPitch() {
	return dotPitch;
}

void Environment::updateDotPitch() {
	switch(dotPitchMethod) {
	case FROM_VIS_ANGLE_SIZE:
		dotPitch = widthm/(double) width;
		pix2deg = width/inputSizeDeg;
		break;
	case FROM_DEG2PIX:
		inputSizeDeg = width/pix2deg;
		widthm = 2*viewingDist*tan(inputSizeDeg/2);
		dotPitch = widthm/(double)width;
		break;
	}
}

double Environment::getInputSizeDeg() {
	return inputSizeDeg;
}

double Environment::getPix2Deg() {
	return pix2deg;
}

double Environment::getViewingDist() {
	return viewingDist;
}

int Environment::getFullHeight() {
	return heightPadded;
}

int Environment::getFullWidth() {
	return widthPadded;
}
