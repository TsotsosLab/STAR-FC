#include "utils.hpp"

void show_mat(const char* window_name, Mat img, int wait) {
	double min, max;
	minMaxLoc(img, &min, &max);
	cout << window_name <<  " min=" << min << " max=" << max << endl;

	Mat to_disp;
	normalize(img, to_disp, 0, 255, NORM_MINMAX, CV_8U);
	namedWindow(window_name, WINDOW_NORMAL);
	imshow(window_name, to_disp);
	waitKey(wait);
}


void printMinMax(const char* img_name, Mat img) {
	double min, max;
	minMaxLoc(img, &min, &max);
	cout << img_name << " min=" << min << " max=" << max << endl;
}

void saveToImg(const char* file_name, Mat img) {
	Mat to_save;
	normalize(img, to_save, 0, 255, NORM_MINMAX, CV_8U);
	imwrite(file_name, to_save);
}

void saveToMat(const char* file_name, Mat img) {
	mat_t *mat;
	matvar_t *matvar;
	size_t dims[2] = {img.rows, img.cols};
	mat = Mat_Create(file_name, NULL);
	if (mat) {
		double *temp_buf = (double *) malloc(img.rows*img.cols*sizeof(double));

		for (int j = 0; j < img.cols; j++) {
			for (int i = 0; i < img.rows; i++) {
				temp_buf[j*img.rows+i] = img.ptr<double>(i)[j];
			}
		}
		matvar = Mat_VarCreate("img", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, temp_buf, 0);
		Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
		free(temp_buf);
		Mat_Close(mat);
	}
}



void saveToMat(const char* file_name, vector<Point> points) {
	mat_t *mat;
	matvar_t *matvar;
	size_t dims[2] = {2, points.size()};
	mat = Mat_Create(file_name, NULL);
	if (mat) {
		double *temp_buf = (double *) malloc(points.size()*2*sizeof(double));

		for (int j = 0; j < points.size(); j++) {
			temp_buf[j*2] = points[j].x;
			temp_buf[j*2+1] = points[j].y;
		}
		matvar = Mat_VarCreate("img", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, temp_buf, 0);
		Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
		free(temp_buf);
		Mat_Close(mat);
	}
}

void readFromMat(const char* file_name, const char* var_name, double* output) {
	int err = 0;
	mat_t *mat;
	matvar_t *matvar;
	mat = Mat_Open(file_name, MAT_ACC_RDONLY);
	if (mat) {
		matvar = Mat_VarRead(mat, var_name);


	}
}

Point swapPointCoords(Point p) {
	return Point(p.y, p.x);
}

// void savePixmap(const char* file_name, QPixmap pixmap) {
// 	QFile file(file_name);
// 	file.open(QIODevice::WriteOnly);
// 	pixmap.save(&file, "PNG");
// 	file.close();
// }



//convolve eye view with kernel and return the result
Mat convolve(Mat img, Mat kernel) {
	//QMutex fftMutex;

	int FFTW_FACTORS[7] = {13,11,7,5,3,2,0}; // end with zero to detect the end of the array

	//fftMutex.lock();
    int h_fft = find_closest_factor(img.rows, FFTW_FACTORS);
    int w_fft = find_closest_factor(img.cols, FFTW_FACTORS);

	float* inputData = (float*) malloc(h_fft*w_fft*sizeof(float));
	float* outputDataPadded = (float*) malloc(h_fft*w_fft*sizeof(float));
	float* kernel_fft = (float*) malloc(h_fft*w_fft*sizeof(float));

	memset(inputData, 0, h_fft*w_fft*sizeof(float));
	memset(kernel_fft, 0, h_fft*w_fft*sizeof(float));

	//copy view into input data
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			inputData[(i%h_fft)*w_fft + (j%w_fft)] += img.ptr<float>(i)[j];
		}
	}

	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			kernel_fft[(i%h_fft)*w_fft+(j%w_fft)] += kernel.ptr<float>(kernel.rows-i-1)[kernel.cols-j-1];
		}
	}

	int complex_width = (w_fft/2.0f)+1;
	int complex_size = complex_width*h_fft;

	//it is necessary to use fftw malloc functions for allocating complex data
	//fftwf_malloc insures word-alignment
	float* image_complex = (float*) fftwf_malloc(h_fft*complex_width*sizeof(fftwf_complex));
	float* kernel_complex = (float*) fftwf_malloc(h_fft*complex_width*sizeof(fftwf_complex));


	//these are the only functions in fftw3 library which are not thread-safe
	//will segfault at runtime if used without a lock in a multi-threaded environment

	//generally it is better to reuse plans instead of generating them every time
	//but since plans depend on the size of the input and output arrays and
	//we cannot guarantee that they will stay the same
	//it is safer to recompute plans for every convolution
	fftwf_plan image_plan = fftwf_plan_dft_r2c_2d(h_fft, w_fft, inputData, (fftwf_complex*)image_complex, FFTW_ESTIMATE); //forward image transform
	fftwf_plan kernel_plan = fftwf_plan_dft_r2c_2d(h_fft, w_fft, kernel_fft, (fftwf_complex*)kernel_complex, FFTW_ESTIMATE); //forward kernel transform
	fftwf_plan inverse_plan = fftwf_plan_dft_c2r_2d(h_fft, w_fft, (fftwf_complex*)kernel_complex, outputDataPadded, FFTW_ESTIMATE); //inverse transform


	//forward fft transform
	fftwf_execute(image_plan);
	fftwf_execute(kernel_plan);

	//compute convolution in the frequency domain
	float *ptr, *ptr2, *ptr_end;
	float re_s, im_s, re_k, im_k;
	for(ptr = image_complex, ptr2 = kernel_complex, ptr_end = image_complex+2*h_fft * (w_fft/2+1); ptr != ptr_end ; ++ptr, ++ptr2)
	{
		re_s = *ptr;
		im_s = *(++ptr);
		re_k = *ptr2;
		im_k = *(++ptr2);
		*(ptr2-1) = re_s * re_k - im_s * im_k;
		*ptr2 = re_s * im_k + im_s * re_k;
	}

	//inverse fft transform for the result
	fftwf_execute(inverse_plan);


	float size = h_fft*w_fft;
	//fftw library doensn't rescale the result
	for (int i = 0; i < size; i++) {
		outputDataPadded[i] =  outputDataPadded[i]/size;
	}


	 //copy the fftw result back to Mat
     Mat output = Mat(img.rows, img.cols, CV_32F);

	 int h_offset = (kernel.rows-1)/2; //these are offsets for the fftw result
	 int w_offset = (kernel.cols-1)/2; //since we are only interested in the valid pixels
								   	   //shift by the size of the kernel
	 for(int x = 0 ; x < output.rows ; x++) {
		for (int y = 0; y < output.cols; y++) {
			output.ptr<float>(x)[y] = outputDataPadded[(x+h_offset)*w_fft+y+w_offset];
		}
	 }

	free(outputDataPadded);
	free(kernel_fft);
	free(inputData);
	fftwf_free(image_complex);
	fftwf_free(kernel_complex);
	fftwf_destroy_plan(kernel_plan);
	fftwf_destroy_plan(image_plan);
	fftwf_destroy_plan(inverse_plan);

	//fftMutex.unlock();
	return output;
}
