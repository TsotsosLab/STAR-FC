/*
 * Author: yulia
 * Date: 18/11/2016
 */
#ifndef EYE_HPP_
#define EYE_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Environment.hpp"
#include "foveate_wide_fov.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

/* Eye class captures a portion of the environment (limited to the defined field of view)
 * at the current fixation.
 */
class Eye {
private:
	Environment* env;
	Foveate_wide_fov* fov;
	Mat view; //current view
	Mat view_fov; //current view with foveation added

	Point currentGazeCoords;

	int width, height;

	bool foveate;


	//TODO: add parameters for foveation...

public:
	//Initialize Eye with the environment
	Eye(Environment* env);
	~Eye();

	//if the image changed
	void reset();

	//get view from the environment
	void viewScene();
	void foveateView();
	void setGazeCoords(Point newGazeCoords);
	Mat getFoveatedView();
	Point getGazeCoords();
};

#endif /* EYE_HPP_ */
