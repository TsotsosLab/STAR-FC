#include "Salicon.hpp"


void showImg(Mat img, char* window_name, int wait) {
		Mat to_disp;
		normalize(img, to_disp, 0, 255, NORM_MINMAX);
		to_disp.convertTo(to_disp, CV_8UC1);
		namedWindow(window_name, WINDOW_AUTOSIZE );
		imshow(window_name, to_disp );
		waitKey(wait);
}

Salicon::~Salicon() {
	delete[] salMapData;
}

Salicon::Salicon() {
	initNet();
	salMapData = NULL;
}

void Salicon::initNet() {
	net = new Net<float>(protoFilename, TEST);
	net->CopyTrainedLayersFrom(caffemodelFilename);

	vector<string> blobNames = net->blob_names();
	vector<string>::iterator it;

	//find index of the saliency map blob in the network
	it=find(blobNames.begin(), blobNames.end(), "saliency_map_out");
	salMapBlobIdx = distance(blobNames.begin(), it);

	it=find(blobNames.begin(), blobNames.end(), "coarse_scale");
	coarseInputBlobIdx = distance(blobNames.begin(), it);

	it=find(blobNames.begin(), blobNames.end(), "fine_scale");
	fineInputBlobIdx = distance(blobNames.begin(), it);

	//Caffe::SetDevice(1);
	Caffe::set_mode(Caffe::GPU);
}

void Salicon::initMeanImg() {
	vector<Mat> channels;
	float mean[] = {103.939f, 116.779f, 123.68f};
	for (int i = 0; i < 3; i++) {
		Mat ch(img.rows, img.cols, CV_32FC1, Scalar(mean[i]));
		channels.push_back(ch);
	}
	merge(channels, meanImg);
}


void Salicon::cropImageToAspectRatio() {
	//crop image to the aspect ratio that corresponds to SALICON training data (600x800 or 1.33)
	int newH, newW;
	int startX, startY;
	//depending on the aspect ratio of the original image
	//crop either horizontally or vertically
	if (img.cols/(double)img.rows > 1.33) {
		newH = img.rows;
		newW = img.rows*800/600;
		startX = 0;
		startY = floor((img.cols-newW)/2);
	} else {
		newH = img.cols*600/800;
		newW = img.cols;
		startX = floor((img.rows-newH)/2);
		startY = 0;
	}
	Mat ROI(img, Rect(startY, startX, newW, newH));
	ROI.copyTo(croppedImg);
//	img.copyTo(croppedImg);
}

/**
 * preprocess the image, create coarse and fine versions and
 * load them into the network inputs
 */

void Salicon::processImage() {
	//preprocess image
	//convert to float and subtract the mean
	img.convertTo(img, CV_32FC3);
	img = img - meanImg;
	img /= 255.0f;

	cropImageToAspectRatio();

	//create coarse and fine input
	resize(croppedImg, coarseImg, Size(800, 600), 0, 0, INTER_NEAREST);
	resize(croppedImg, fineImg, Size(1600, 1200), 0, 0, INTER_NEAREST);

	//copy images to the input layer of the CNN

	const boost::shared_ptr<Blob<float> > coarseInputBlob = net->blobs()[coarseInputBlobIdx];
	const boost::shared_ptr<Blob<float> > fineInputBlob= net->blobs()[fineInputBlobIdx];


	int num_channels = coarseInputBlob->channels();
	  CHECK(num_channels == 3)
	    << "Input layer should have 3 channels.";

	int width = coarseInputBlob->width();
	int height = coarseInputBlob->height();
	coarseInputChannels.clear();
	float* coarseInputData = coarseInputBlob->mutable_cpu_data();
	for (int i = 0; i < coarseInputBlob->channels(); ++i) {
		Mat channel(height, width, CV_32FC1, coarseInputData);
		coarseInputChannels.push_back(channel);
		coarseInputData += width * height;
	}

	width = fineInputBlob->width();
	height = fineInputBlob->height();
	fineInputChannels.clear();
	float* fineInputData = fineInputBlob->mutable_cpu_data();
	for (int i = 0; i < fineInputBlob->channels(); ++i) {
		Mat channel(height, width, CV_32FC1, fineInputData);
		fineInputChannels.push_back(channel);
		fineInputData += width * height;
	}

	split(coarseImg, coarseInputChannels);
	split(fineImg, fineInputChannels);

	CHECK(reinterpret_cast<float*>(coarseInputChannels.at(0).data)
	        == net->blobs()[coarseInputBlobIdx]->cpu_data())
	<< "ERROR: Coarse input channels are not wrapping the input layer of the network.";
	CHECK(reinterpret_cast<float*>(fineInputChannels.at(0).data)
	        == net->blobs()[fineInputBlobIdx]->cpu_data())
	<< "ERROR: Fine input channels are not wrapping the input layer of the network.";
}

/**
 * Read image
 */
void Salicon::loadImage(string imgName) {
	img = imread(imgName.c_str(), CV_LOAD_IMAGE_COLOR);
	initMeanImg();
	processImage();
}


void Salicon::loadImage(Mat new_img) {
	new_img.copyTo(this->img);
	img *= 255.0; //multiply by 255 before subtracting the mean
	initMeanImg();
	processImage();
}

Mat Salicon::computeSalMap() {

	net->Forward();

	Blob<float> output_blob(net->blobs()[salMapBlobIdx]->shape());
	output_blob.CopyFrom(*net->blobs()[salMapBlobIdx], false, false);

	const vector<int> shape = output_blob.shape();
	vector<int> new_shape;
	new_shape.push_back(shape[2]);
	new_shape.push_back(shape[3]);

	output_blob.Reshape(new_shape);
	const float* outputBlobData = output_blob.cpu_data();

	if (!salMapData) {
		salMapData = new float[new_shape[0]*new_shape[1]];
	}
	memcpy(salMapData, outputBlobData, new_shape[0]*new_shape[1]*sizeof(float));

	Mat salMapRaw(new_shape[0], new_shape[1], CV_32FC1, salMapData);

	resize(salMapRaw, salMap, Size(croppedImg.cols, croppedImg.rows), 0, 0, INTER_CUBIC);


	//pad cropped saliency map with zeros to the original dimensions of the image
	double vPad = (img.rows - croppedImg.rows)/2.0;
	double hPad = (img.cols - croppedImg.cols)/2.0;
	copyMakeBorder(salMap, salMap, floor(vPad), ceil(vPad), floor(hPad), ceil(hPad), BORDER_CONSTANT, Scalar(0));

	normalize(salMap, salMap, 0.0, 1.0, NORM_MINMAX);

	//showImg(salMap, "SalMap", 1);
	return salMap;
}
