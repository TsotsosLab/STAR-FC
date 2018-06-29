/*
 * Author: yulia
 * Date: 18/11/2016
 */
#include "CentralAttentionalMap.hpp"


CentralAttentionalMap::CentralAttentionalMap(int h, int w, double deg2pix, double cSizeDeg) {
	this->height = h;
	this->width = w;
	this->deg2pix = deg2pix;
	this->centralFieldSizeDeg = cSizeDeg;
	centralFieldSizePix = centralFieldSizeDeg/deg2pix;
	initCentralMask(h, w);
#ifdef WITH_SALICON
	salicon = new Salicon();
#endif
}

CentralAttentionalMap::~CentralAttentionalMap() {

}

void CentralAttentionalMap::reset(int h, int w) {
	height = h;
	width = w;
	initCentralMask(h, w);
}

void CentralAttentionalMap::loadFaceTemplates() {
	/* This code is adapted from test_mat.c code shipped with MatIO library
	 * see function test_readslab()
	 */
	mat_t *mat;
	matvar_t *matvar;
	int start[2] = {0, 0};
	int stride[2] = {1, 1};
	int edge[2] = {2, 2};
	double* data_ptr;

	int err;
	//templates in face_templates are in BGR order
	mat = Mat_Open("../data/face_templates.mat", MAT_ACC_RDONLY);
	if (mat) {
		while (NULL != (matvar = Mat_VarReadNextInfo(mat))) {
			//Mat_VarPrint(matvar, 1);
			edge[0] = matvar->dims[0];
			edge[1] = matvar->dims[1];

			data_ptr = new double[matvar->dims[0]*matvar->dims[1]];
			err = Mat_VarReadData(mat, matvar, data_ptr, start, stride, edge);
			if (err) {
				cout << "Could not read variable from .mat file!" << endl;
			}

			Mat temp = Mat(matvar->dims[0], matvar->dims[1], CV_64F, data_ptr);
			transpose(temp, temp); //matlab stores data in column major order and MatIO does not correct
								//for it, so data is transposed later
			Mat temp1;
			temp.copyTo(temp1);
			faceTemplates.push_back(temp1);

			Mat_VarFree(matvar);
			delete[] data_ptr;

		}
		Mat_Close(mat);
	} else {
		cout << "Could not open Mat file\n" << endl;
	}
}

//create central mask with radius equal to centralFieldSizePix
//all pixels within the radius are set to 1 and 0 otherwise
void CentralAttentionalMap::initCentralMask(int h, int w){
	centralMask = Mat(h, w, CV_8U);

	int centX = h/2;
	int centY = w/2;
	double radius;

	for(int i=0; i < h; i++){
		for(int j=0; j < w; j++){
			radius = sqrt((double)((i-centX)*(i-centX) + (j-centY)*(j-centY)));
			if(radius <= centralFieldSizePix){
				centralMask.at<char>(i,j) = 1;
			}else{
				centralMask.at<char>(i,j) = 0;
			}
		}
	}
}

#ifdef WITH_SALICON
void CentralAttentionalMap::centralDetection(Mat view) {
	Mat saveImg;
	view.copyTo(saveImg);
	normalize(saveImg, saveImg, 0.0, 255.0, NORM_MINMAX);
	saveImg.convertTo(saveImg, CV_8U);
	salicon->loadImage(view);
	centralSaliencyMap = salicon->computeSalMap();
	centralSaliencyMap.convertTo(centralSaliencyMap, CV_64FC1);

	//saveToImg("output/apples/central_map.png", centralSaliencyMap);
}
#endif

//detect faces in the eye view
void CentralAttentionalMap::detectFaces(Mat view) {
	//convolve eye view with each face template
	Mat viewBGR[3];
	split(view, viewBGR);
	double minVal, maxVal, globalMinVal, globalAvg;
	double avg;

	Mat faceCombinedBGR[3];
	vector<Mat> faceChannels;
	Mat out;
	for(int i = 0; i < faceTemplates.size()-1; i+=3) {
		avg = 0;
		globalMinVal = numeric_limits<double>::max();
		for (int j = 0; j < 3; j++) {
			//Mat out = convolve(viewBGR[j], faceTemplates[i+j]);
			filter2D(viewBGR[j], out, -1, faceTemplates[i+j], Point(-1, -1), 0, BORDER_CONSTANT);
			//printMinMax("FaceTemplate", out);
			minMaxLoc(out, &minVal, NULL, NULL, NULL);
			globalMinVal = min(globalMinVal, minVal);
			out.copyTo(faceCombinedBGR[j]);
			avg += mean(out).val[0];
		}

		Mat faceCombined = Mat(view.rows, view.cols, CV_64F, Scalar(0));;

		globalAvg = avg - globalMinVal;
		for (int j = 0; j < 3; j++) {
			faceCombined += (faceCombinedBGR[j]-globalMinVal)/globalAvg;
		}
		faceChannels.push_back(faceCombined);
		//printMinMax("faceCombined", faceCombined);
	}

	faceDetectionResult = Mat(view.rows, view.cols, CV_64F, Scalar(0));
	for (int i = 0; i < faceChannels.size(); i++) {
		max(faceDetectionResult, faceChannels[i], faceDetectionResult);
	}
	printMinMax("faceDetectionResult", faceDetectionResult);
}

//apply central mask to the output of face detection
void CentralAttentionalMap::maskFaceDetection() {
	faceDetectionResult.copyTo(centralMap, centralMask);
	printMinMax("centralAttentionalMap", centralMap);

	//normalize(centralMap, centralMap, 0, 1, NORM_MINMAX, CV_64F);
	//show_mat("Central Attentional Map", centralMap, 0);
}

#ifdef WITH_SALICON
void CentralAttentionalMap::maskCentralDetection() {
	centralSaliencyMap.copyTo(centralMap, centralMask);
}
#endif

Mat CentralAttentionalMap::getCentralMap() {
	return centralMap;
}

double CentralAttentionalMap::getCentralMapMax() {
	double maxVal;
	minMaxLoc(centralMap, NULL, &maxVal, NULL, NULL);
	return maxVal;
}


