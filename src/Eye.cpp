
/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "Eye.hpp"

Eye::Eye(Environment* env){
	this->currentGazeCoords = Point(-1, -1);
	this->env = env;
	this->foveate = true;
	this->height = env->getHeight();
	this->width = env->getWidth();
	this->fov = new Foveate_wide_fov();
}

Eye::~Eye(){

}

void Eye::reset() {
	currentGazeCoords = Point(-1, -1);
	height = env->getHeight();
	width = env->getWidth();
	//view = Scalar(0);
	//view_fov = Scalar(0);
}

//get view from the environment
//according to the current fixation
void Eye::viewScene(){

	if (currentGazeCoords.x < 0 || currentGazeCoords.y < 0) {
		currentGazeCoords = Point(width/2, height/2);
	}

	view = env->getEyeView(currentGazeCoords, width, height);
	if (foveate) {
		foveateView();
	}
//	Mat temp = fov->foveate(env->getScene(), Point(800, 420), env->getDotPitch(), env->getViewingDist(), true);
//	saveToImg("output/apples/foveated_center.jpg", temp);
//
//	temp = fov->foveate(env->getScene(), Point(200, 800), env->getDotPitch(), env->getViewingDist(), true);
//	saveToImg("output/apples/foveated_off_center.jpg", temp);
//	//show_mat("foveated", view_fov, 1);
}

void Eye::foveateView(){
//    Mat view_cones_rods = fov->foveate(view, Point(width/2, height/2), env->getDotPitch(), env->getViewingDist(), true);
//    drawMarker(view_cones_rods, Point(width/2, height/2), Scalar(0, 0, 1), MARKER_CROSS, 30, 4);
//    saveToImg("/home/yulia/Documents/ONR-TD/documentation/conference_paper/arXiv/images/11_fov_cones_rods.jpg", view_cones_rods);
//    Mat view_cones_only = fov->foveate(view, Point(width/2, height/2), env->getDotPitch(), env->getViewingDist(), false);
//    drawMarker(view_cones_only, Point(width/2, height/2), Scalar(0, 0, 1), MARKER_CROSS, 30, 4);
//    saveToImg("/home/yulia/Documents/ONR-TD/documentation/conference_paper/arXiv/images/11_fov_cones_only.jpg", view_cones_only);
	view_fov = fov->foveate(view, Point(width/2, height/2), env->getDotPitch(), env->getViewingDist(), true);
}

Mat Eye::getFoveatedView() {
	return view_fov;
}

void Eye::setGazeCoords(Point newGazeDirection) {
	currentGazeCoords += newGazeDirection;
	currentGazeCoords.x = max(0, min(width-1, currentGazeCoords.x));
	currentGazeCoords.y = max(0, min(height-1, currentGazeCoords.y));

}

Point Eye::getGazeCoords() {
	return currentGazeCoords;
}
