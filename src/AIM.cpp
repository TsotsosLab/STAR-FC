
#include "AIM.h"

AIM::AIM() {
	orig_h = -1;
	orig_w = -1;
}

AIM::~AIM() {
	for (int i = 0; i < num_kernels; i++) {
		delete [] kernels[i];
	}
	delete [] kernels;
	delete [] kernel_data;

	aim_temp.clear();

}

/* Load basis from a binary file
 * Expects the binary file to be formatted as follows:
 * first 3 floats are number of kernels, kernel_size
 * and number of channels in the image (1 for grayscale and 3 for rgb)
 * followed by the num_channels*num_kernels*kernel_size*kernel_size of floats
 * containing the basis written in a row-major order*/
void AIM::loadBasis(char* filename) {
	FILE* kernel_file;
	kernel_file = fopen(filename, "rb");
	float temp;

	fread(&temp, sizeof(float), 1, kernel_file); num_kernels = temp;
	fread(&temp, sizeof(float), 1, kernel_file); kernel_size = temp;
	fread(&temp, sizeof(float), 1, kernel_file); num_channels = temp;

	kernels = new Mat*[num_kernels];
	for (int i = 0; i < num_kernels; i++) {
		kernels[i] = new Mat[num_channels];
	}

	kernel_data = new float[num_channels*num_kernels*kernel_size*kernel_size];

	//read all data into array of floats
	fread(kernel_data, sizeof(float), num_kernels*kernel_size*kernel_size*num_channels, kernel_file);
	int sizes[] = {kernel_size, kernel_size, num_channels};
	printf("Found %i kernels DIM %i x %i x %i\n", num_kernels, kernel_size, kernel_size, num_channels);
	printf("Loading kernels...\n");
	double min, max;
	//load data into Mat
	for (int c = 0; c < num_channels; c++) {
		for (int n = 0; n < num_kernels; n++) {
			kernels[n][c] = Mat(kernel_size, kernel_size, CV_32FC1, &kernel_data[c*num_kernels*kernel_size*kernel_size+n*kernel_size*kernel_size]);
			minMaxLoc(kernels[n][c], &min, &max);
		}
	}

	fclose(kernel_file);

	convolver = cuda::createConvolution(Size(kernel_size, kernel_size));
}


void AIM::update() {
	if (aim_temp.size() > 0) {
		if (image.rows == orig_h && image.cols == orig_w) {
			return;
		} else {
			aim_temp.clear();
		}
	}
	for (int f = 0; f < num_kernels; f++) {
		aim_temp.push_back(Mat(image.rows-kernel_size+1, image.cols-kernel_size+1, CV_32FC1));
		//aim_temp.push_back(Mat(image.rows, image.cols, CV_32FC1));
	}

}

void AIM::loadImage(Mat image, float scale_factor) {
	image.copyTo(this->image);

	update();

	orig_h = image.rows;
	orig_w = image.cols;
	//this->scale = scale_factor;
	//resize(this->image, this->image, cvSize(0, 0), scale, scale, CV_INTER_AREA);

	new_w = 700;
	new_h = image.rows*((double) new_w/image.cols);

	resize(this->image, this->image, Size(new_w, new_h), 0, 0, CV_INTER_AREA);

}

/* load image for processing
 * filename - full path to an image
 * scale - scaling factor, 0.5 is recommended
 */
void AIM::loadImage(const char* filename, float scale_factor) {
	image = imread(filename, CV_LOAD_IMAGE_COLOR);

	update();

	orig_h = image.rows;
	orig_w = image.cols;
	printf("Loaded %s \n Image size: %i x %i\n", filename, image.rows, image.cols);
	scale = scale_factor;

	new_w = 700;
	new_h = image.rows*((double) new_w/image.cols);

	resize(this->image, this->image, Size(new_w, new_h), 0, 0, CV_INTER_AREA);
	//resize(image, image, cvSize(0, 0), scale, scale);

}

/* run AIM saliency algorithm on the image
 */
void AIM::runAIM() {

	min_aim = 100000;
	max_aim = -1000000;
	int border = kernel_size/2;

	//split image into channels
	split(image, channels);


	for (int c = 0; c < num_channels; c++) {
		channels[c].convertTo(channels[c], CV_32FC1);
//		string fname = "/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/aim_img"+to_string(c)+".mat";
//		saveToMat(fname.c_str(),  channels[c]);
	}

//	chrono::time_point<chrono::system_clock> start, end;
//	start = chrono::system_clock::now();


	//apply all filters to each channel
	Point anchor(-1, -1);
	for (int f = 0; f < num_kernels; f++) {
		//filter2D(channels[0], aim_temp[f], -1, kernels[f][0], anchor, 0, BORDER_CONSTANT);
		//cuda::copyMakeBorder(channels[0], temp1, border, border, border, border, BORDER_CONSTANT, Scalar(0));
		convolver->convolve(channels[2], kernels[f][0], aim_temp[f], true);

//		string fname = "/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/aim_temp" + to_string(f) + "_0.mat";
//		saveToMat(fname.c_str(),  aim_temp[f]);
//
//		fname = "/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/kernels" + to_string(f) + "_0.mat";
//		saveToMat(fname.c_str(),  kernels[f][0]);

		for(int c = 1; c < num_channels; c++) {
			//filter2D(channels[c], temp, -1, kernels[f][c], anchor, 0, BORDER_CONSTANT);
			//cuda::copyMakeBorder(channels[c], temp1, border, border, border, border, BORDER_CONSTANT, Scalar(0));
			convolver->convolve(channels[c], kernels[f][c], temp, true);
			aim_temp[f] += temp;

//			fname = "/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/aim_temp" + to_string(f) + "_" + to_string(c) + ".mat";
//			saveToMat(fname.c_str(), temp);
//
//			fname = "/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/kernels" + to_string(f) + "_" + to_string(c) + ".mat";
//			saveToMat(fname.c_str(),  kernels[f][c]);
		}
		//only keep the valid pixels after filtering
		//aim_temp[f] = aim_temp[f].colRange(border, image.cols - border)
		//						 .rowRange(border, image.rows - border);
		minMaxLoc(aim_temp[f], &min, &max);

		//compute max and min across all feature maps
		max_aim = fmax(max, max_aim);
		min_aim = fmin(min, min_aim);
	}

	printf("max_aim=%f min_aim=%f\n", max_aim, min_aim);
//	end = chrono::system_clock::now();
//	chrono::duration<double> elapsedSec = end-start;
//	cout << "convolution took " << elapsedSec.count() << " sec" << endl;

	//rescale image using global max and min
	for (int f = 0; f < num_kernels; f++) {
		aim_temp[f] -= min_aim;
		aim_temp[f] /= (max_aim - min_aim);
	}


	//compute histograms for each feature map
	//and use them to rescale values based on histogram to reflect likelihood
	Mat hist;
	Mat sm = Mat(aim_temp[0].rows, aim_temp[0].cols, CV_32FC1, Scalar::all(0));;
	float div = 1.0/(aim_temp[0].rows*aim_temp[0].cols) ;
	float histRange[] = {0, 1};
	const float *range[] = {histRange};
	int histSize[] = {256};

	Mat idx;
	for (int f = 0; f < num_kernels; f++) {
		calcHist(&aim_temp[f], 1, 0, Mat(), hist, 1, histSize, range, true, false);
		idx = aim_temp[f] * (histSize[0]-1);
		for(int i = 0; i < aim_temp[f].rows; i++) {
			for (int j = 0; j < aim_temp[f].cols; j++) {
				//find index of the value in the histogram
				int idx = round(aim_temp[f].at<float>(i, j) * (histSize[0]-1));
				//compute log probability
				sm.at<float>(i,j) -= log(hist.at<float>(idx)*div+0.000001f);
			}
		}
	}

	normalize(sm, adj_sm, 0.0, 1.0, NORM_MINMAX, CV_64F);

	//add blank border
	copyMakeBorder(adj_sm, adj_sm, border, border, border, border, BORDER_CONSTANT, 0);

	//blur
	GaussianBlur(adj_sm, adj_sm, Size(31, 31), 8, 8, BORDER_CONSTANT);

	//rescale image back to the original size
	resize(adj_sm, adj_sm, cvSize(orig_w, orig_h), -1 , -1);
}

/* save final saliency map
 * filename - full path to the file
 */
void AIM::saveResult(const char* filename) {
	imwrite(filename, adj_sm);
}
