/*
Author: yulia
Date: 1/04/2016
*/

#include "foveate_wide_fov.hpp"

Foveate_wide_fov::Foveate_wide_fov() {
    CTO = 1/64.0;    // constant from Geisler&Perry
    alpha = (0.106)*1; 			// constant from Geisler&Perry
    epsilon2 = 2.3; 			  // constant from Geisler&Perry
    orig_w = -1;
    orig_h = -1;

}

Foveate_wide_fov::~Foveate_wide_fov() {
	cleanup();
}

void Foveate_wide_fov::setImage(Mat input_img) {
	input_img.copyTo(this->input_img);
}

void Foveate_wide_fov::init() {
	orig_h = input_img.rows;
	orig_w = input_img.cols;

	num_levels = min(7.0, floor(log2(std::max(input_img.rows, input_img.cols))));

	output_img = Mat(input_img.rows, input_img.cols, CV_DATA_T, Scalar(0));
	for (int ch = 0; ch < 3; ch++) {
		output[ch] = Mat(input_img.rows, input_img.cols, CV_DATA_T, Scalar(0));
	}
	//--------------allocate all arrays--------------------

	xi_buf = (double *) malloc(input_img.rows*input_img.cols*sizeof(double));
	yi_buf = (double *) malloc(input_img.rows*input_img.cols*sizeof(double));


	fov_cones_buf = (data_t *) malloc(input_img.rows*input_img.cols*sizeof(data_t));
	fov_rods_buf = (data_t *) malloc(input_img.rows*input_img.cols*sizeof(data_t));

	pyrlevel_cones_buf = (data_t *) malloc(input_img.rows*input_img.cols*sizeof(data_t));
	pyrlevel_rods_buf = (data_t *) malloc(input_img.rows*input_img.cols*sizeof(data_t));
	pyrlevel_cones = Mat(input_img.rows, input_img.cols, CV_64F, Scalar(0));
	pyrlevel_rods = Mat(input_img.rows, input_img.cols, CV_64F, Scalar(0));


	pyramid_buf = (data_t*) malloc(input_img.rows*input_img.cols*num_levels*sizeof(data_t));
	for (int i = 0; i < num_levels; i++) {
	//pyramid[i] = Mat(input_img.rows, input_img.cols, CV_DATA_T, Scalar(0));
	  pyramid.push_back(Mat(input_img.rows, input_img.cols, CV_DATA_T, &pyramid_buf[i*input_img.rows*input_img.cols]));
	}


	ec = Mat(input_img.rows, input_img.cols, CV_64F, Scalar(0));
	ex = Mat(input_img.rows, input_img.cols, CV_64F, Scalar(0));
	ey = Mat(input_img.rows, input_img.cols, CV_64F, Scalar(0));
	//---------------------------------------------

}


void Foveate_wide_fov::cleanup() {
	//cleanup
	free(xi_buf);
	free(yi_buf);
	free(pyramid_buf);
	free(pyrlevel_cones_buf);
	free(pyrlevel_rods_buf);
	free(fov_cones_buf);
	free(fov_rods_buf);
	pyramid.clear();
}

void Foveate_wide_fov::preprocess(Mat input_img, Point gaze_pos, double dotpitch, double viewingdist, bool rods_and_cones) {

	  compute_meshgrid();

	  ex = ex - gaze_pos.x;
	  ey = ey - gaze_pos.y;

	  // eradius is the radial distance between each point and the point
	  // of gaze.  This is in meters.
	   pow(ex, 2, ex);
	   pow(ey, 2, ey);
	   eradius = ex + ey;
	   sqrt(eradius, dist_px); //distance from gaze point in pixels
	   eradius = dist_px * dotpitch;


	   // calculate ec, the eccentricity from the foveal center, for each
	   // point in the image.  ec is in degrees.
	   // no per-element atan function for Mat in OpenCV
	   //so do a loop
	   for (int i = 0; i < input_img.rows; i++) {
		 for (int j = 0; j < input_img.cols; j++) {
		   //round up all values below 7 to increase size of fovea
		   //ec.ptr<double>(i)[j] = max(7.0, 180.0*atan(eradius.ptr<double>(i)[j]/viewingdist)/M_PI);
			 ec.ptr<double>(i)[j] = 180.0*atan(eradius.ptr<double>(i)[j]/viewingdist)/M_PI;
		 }
	   }

	  eyefreq_cones = alpha*(ec+epsilon2);
	  eyefreq_cones = epsilon2 / eyefreq_cones;
	  //pow(eyefreq_cones, 5, eyefreq_cones);
	  eyefreq_cones = eyefreq_cones * log(1/CTO);
	  pow(eyefreq_cones, 0.3, eyefreq_cones);
	  minMaxLoc(eyefreq_cones, &min_val, &max_val, &p_min, &p_max);
	  eyefreq_cones = (eyefreq_cones-min_val)/(max_val-min_val);

	  // pyrlevel is the fractional level of the pyramid which must be
	  // used at each pixel in order to match the foveal resolution
	  // function defined above.
	  //pyrlevel = maxfreq ./ eyefreq;
	  //divide(maxfreq, eyefreq_cones, pyrlevel_cones);
	  eyefreq_cones = 1 - eyefreq_cones;
	  pyrlevel_cones = ((double)(num_levels-1))*eyefreq_cones;

	  // constrain pyrlevel in order to conform to the levels of the
	  // pyramid which have been computed.
	  for (int i = 0; i < input_img.rows; i++) {
		for (int j = 0; j < input_img.cols; j++) {
		  pyrlevel_cones.ptr<double>(i)[j] = std::max(0.0, std::min((double) num_levels, pyrlevel_cones.ptr<double>(i)[j]));
		}
	  }

//	  saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/pyrlevel_cones.mat", pyrlevel_cones);
	  if (rods_and_cones) {

	    double p[6] = {8.8814e-11,  -1.6852e-07,   1.1048e-04,  -3.1856e-02,   3.7501e+00,  -3.0283e+00};
	    eyefreq_rods = polyval(dist_px, p, 5);
//	    saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/dist_px.mat", dist_px);
//	    saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/eyefreq_rods.mat", eyefreq_rods);

	    //eyefreq_rods = ggamma(ec, 2.46, 50*1.74, 0.77, 35*15.1, -1)-2;
	    minMaxLoc(eyefreq_rods, &min_val, &max_val, &p_min, &p_max);

	    eyefreq_rods = eyefreq_rods / max_val;
		eyefreq_rods = 1 - eyefreq_rods;
		pyrlevel_rods = num_levels*eyefreq_rods;

		//pyrlevel_rods(pyrlevel_rods < 0) = num_levels;
		//pyrlevel_rods = max(1, min(levels, pyrlevel_rods));

		for (int i = 0; i < input_img.rows; i++) {
		  for (int j = 0; j < input_img.cols; j++) {
			  //if (pyrlevel_rods.ptr<double>(i)[j] > 0) {
				pyrlevel_rods.ptr<double>(i)[j] = std::max(0.0, std::min((double) num_levels, pyrlevel_rods.ptr<double>(i)[j]+2));
			  //} else {
			 //	  pyrlevel_rods.ptr<double>(i)[j] = std::max(0.0, pyrlevel_rods.ptr<double>(i)[j]);
			  //}
		  }
		}

	  }
	  //meshgrid(input_img, xi, yi);
	  //saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/pyrlevel_rods.mat", pyrlevel_rods);

	  for (int i = 0; i < input_img.rows; i++) {
		for (int j = 0; j < input_img.cols; j++) {
		  xi_buf[i*input_img.cols+j] = i;
		  yi_buf[i*input_img.cols+j] = j;
		  pyrlevel_cones_buf[i*input_img.cols+j] = pyrlevel_cones.ptr<double>(i)[j];
		  if (rods_and_cones) {
			pyrlevel_rods_buf[i*input_img.cols+j] = pyrlevel_rods.ptr<double>(i)[j];
		  }
		}
	  }
}

Mat Foveate_wide_fov::foveate(Mat input_img, Point gaze_pos, double dotpitch, double viewingdist, bool rods_and_cones) {
  if (orig_h < 0 || orig_w < 0) {
	  setImage(input_img);
	  init();
  } else {
	  if (orig_h != input_img.rows || orig_w != input_img.cols) {
		  cleanup();
		  input_img.copyTo(this->input_img);
		  init();
	  }
  }
//  saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/input_img.mat", input_img);
  //------------------- compute foveation ----------
	chrono::time_point<chrono::system_clock> start, end;
	start = chrono::system_clock::now();

//#ifndef GPU_FOVEATION

//	data_t* pyrlevel_cones_buf =(data_t *) malloc(input_img.rows*input_img.cols*sizeof(data_t));
//	data_t* pyrlevel_rods_buf =(data_t *) malloc(input_img.rows*input_img.cols*sizeof(data_t));

	// preprocess_GPU(input_img.rows, input_img.cols, gaze_pos.y, gaze_pos.x, num_levels, //we assume that x is vertical and y is horizontal
	// 				CTO, alpha, epsilon2, dotpitch, viewingdist,						//but Point in OpenCV is the opposite
	// 				pyrlevel_cones_buf, pyrlevel_rods_buf, rods_and_cones);
//#else

	preprocess(input_img, gaze_pos, dotpitch, viewingdist, rods_and_cones);

	//this is some code for testing below, will do proper testing later
//	double mean_sq_error = 0;
//	for (int i = 0; i <input_img.rows*input_img.cols; i++) {
//		mean_sq_error += (pyrlevel_cones_gpu[i] - pyrlevel_cones_buf[i]) * (pyrlevel_cones_gpu[i] - pyrlevel_cones_buf[i]);
//	}
//	mean_sq_error /= input_img.rows*input_img.cols;
//	cout << "pyrlevel cones mse = " << mean_sq_error << " " << (mean_sq_error < 0.0001) << endl;


//	if (rods_and_cones) {
//		mean_sq_error = 0;
//		for (int i = 0; i < input_img.rows*input_img.cols; i++) {
//			mean_sq_error += (pyrlevel_rods_gpu[i] - pyrlevel_rods_buf[i]) * (pyrlevel_rods_gpu[i] - pyrlevel_rods_buf[i]);
//		}
//		mean_sq_error /= input_img.rows*input_img.cols;
//		cout << "pyrlevel rods mse = " << mean_sq_error << " " << (mean_sq_error < 0.0001) << endl;
//
//	}

//#endif

	end = chrono::system_clock::now();
	chrono::duration<double> elapsedSec = end-start;
	//cout << "preprocessing took " << elapsedSec.count() << " sec" << endl;


  if (input_img.channels() == 1) {
	input_img.convertTo(input_img, CV_DATA_T);

	start = chrono::system_clock::now();
	compute_image_pyramid_alloc(input_img);
	end = chrono::system_clock::now();
	chrono::duration<double> elapsedSec = end-start;
	//cout << "image pyramigd took " << elapsedSec.count() << " sec" << endl;

#ifdef GPU_FOVEATION
		ba_interp3_GPU(pyramid_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#else
		//interpolate between levels of pyramid
		ba_interp3(pyramid_buf, xi_buf, yi_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#endif
		for (int i = 0; i < input_img.rows; i++) {
		  for (int j = 0; j < input_img.cols; j++) {
			output_img.ptr<data_t>(i)[j] = fov_cones_buf[i*input_img.cols + j];
		  }
		}

		if (rods_and_cones) {
#ifdef GPU_FOVEATION
			ba_interp3_GPU(pyramid_buf, pyrlevel_rods_buf, fov_rods_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#else
		//interpolate between levels of pyramid
			ba_interp3(pyramid_buf, xi_buf, yi_buf, pyrlevel_rods_buf, fov_rods_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#endif
			for (int i = 0; i < input_img.rows; i++) {
			  for (int j = 0; j < input_img.cols; j++) {
				output_img.ptr<data_t>(i)[j] = 0.7*fov_cones_buf[i*input_img.cols + j] + 0.3*fov_rods_buf[i*input_img.cols + j];
			  }
			}
		}
  } else {
	//for color images compute transform for each channel separately
	//and combine them for the final result

	if (rods_and_cones) {
		cvtColor(input_img, input_img, COLOR_BGR2YCrCb);
	}

	split(input_img, channels);

	for (int ch = 0; ch < 3; ch++) {

		channels[ch].convertTo(channels[ch], CV_DATA_T);

		start = chrono::system_clock::now();
		compute_image_pyramid_alloc(channels[ch]);

		end = chrono::system_clock::now();
		elapsedSec = end-start;
		//cout << "image pyramid took " << elapsedSec.count() << " sec" << endl;

		if (~rods_and_cones) {
#ifdef GPU_FOVEATION
			ba_interp3_GPU(pyramid_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#else
			//interpolate between levels of pyramid
			ba_interp3(pyramid_buf, xi_buf, yi_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#endif
			//copy results
			for (int i = 0; i < input_img.rows; i++) {
				for (int j = 0; j < input_img.cols; j++) {
					output[ch].ptr<data_t>(i)[j] = fov_cones_buf[i*input_img.cols + j];
				}
			}

		//compute rods only for the intensity channel and combine with corresponding cones output
		} else {
			if (ch == 0) {

#ifdef GPU_FOVEATION
			ba_interp3_GPU(pyramid_buf, pyrlevel_rods_buf, fov_rods_buf, input_img.rows, input_img.cols, num_levels, "cubic");
			//ba_interp3_GPU(pyramid_buf, xi_buf, yi_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#else
			ba_interp3(pyramid_buf, xi_buf, yi_buf, pyrlevel_rods_buf, fov_rods_buf, input_img.rows, input_img.cols, num_levels, "cubic");
			ba_interp3(pyramid_buf, xi_buf, yi_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#endif
			for (int i = 0; i < input_img.rows; i++) {
			  for (int j = 0; j < input_img.cols; j++) {
				output[ch].ptr<data_t>(i)[j] = 0.7*output[ch].ptr<data_t>(i)[j] + 0.3*fov_rods_buf[i*input_img.cols + j];
			  }
			}

		// compute only cones for the color channels
			} else {
#ifdef GPU_FOVEATION
				ba_interp3_GPU(pyramid_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#else
				ba_interp3(pyramid_buf, xi_buf, yi_buf, pyrlevel_cones_buf, fov_cones_buf, input_img.rows, input_img.cols, num_levels, "cubic");
#endif
				for (int i = 0; i < input_img.rows; i++) {
					for (int j = 0; j < input_img.cols; j++) {
						output[ch].ptr<data_t>(i)[j] = fov_cones_buf[i*input_img.cols + j];
					}
				}
			}
		}
	}


	//combine color channels into a single image
	merge(output, 3, output_img);
  }
//
//  saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/foveated0.mat", output[0]);
//  saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/foveated1.mat", output[1]);
//  saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/foveated2.mat", output[2]);

  if (rods_and_cones) {
	output_img.convertTo(output_img, CV_8U);
	//conversion is not implemented for doubles so convert first
	cvtColor(output_img, output_img, COLOR_YCrCb2BGR);

	input_img.convertTo(input_img, CV_8U);
	cvtColor(input_img, input_img, COLOR_YCrCb2BGR);
  }

  output_img.convertTo(output_img, CV_64F); //convert back to double

//  split(output_img, channels);
//
//    saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/foveatedRodsCones0.mat", channels[0]);
//    saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/foveatedRodsCones1.mat", channels[1]);
//    saveToMat("/home/yulia/Documents/ONR-TD/fixation_code/STAR-FCpy/foveatedRodsCones2.mat", channels[2]);


  normalize(output_img, output_img, 0, 1, NORM_MINMAX, CV_64F);




//#ifdef GPU_FOVEATION
//         saveToImg("/home/yulia/Desktop/foveated_CUDA.png", output_img);
//         saveToMat("/home/yulia/Desktop/foveated_CUDA.mat", output_img);
//#else
//         saveToImg("/tmp/foveated.png", output_img);
//         saveToMat("/tmp/foveated.mat", output_img);
//
//#endif

  return output_img;
}




/*
  * Compute image pyramid assuming that memory for pyramid was allocated manually.
  */
void Foveate_wide_fov::compute_image_pyramid_alloc(Mat img) {

	    Mat tmp, tmp1;
	    double ratio = 0.5;
	    img.copyTo(pyramid[0]);
	    img.copyTo(tmp);
	    for (int i = 1; i < num_levels; i++) {
	    	pyrDown(tmp, tmp1, Size(round(tmp.cols*ratio), round(tmp.rows*ratio)), BORDER_DEFAULT);
	    	resize(tmp1, pyramid[i], Size(img.cols, img.rows), 0, 0, INTER_LINEAR);
	    	tmp1.copyTo(tmp);
	    	//img.copyTo(pyr[i]);

//			printf("pyramid %i\n", i);
//			for (int i = 0; i < 5; i++) {
//				for (int j = 0; j < 5; j++) {
//					printf("%0.03f ", pyramid[1].ptr<data_t>(i)[j]);
//				}
//				printf("\n");
//			}

	    }


 }

 //same as Matlab function
void Foveate_wide_fov::compute_meshgrid() {
     for (int i = 0; i < input_img.rows; i++) {
       for (int j = 0; j < input_img.cols; j++) {
         ex.ptr<double>(i)[j] = j;
         ey.ptr<double>(i)[j] = i;
       }
     }
 }

 //compute generalized gamma distribution for vector X
Mat Foveate_wide_fov::ggamma(Mat X, double alpha, double beta, double g, double sigma, double mu)  {

   //y = sigma*((g*exp(-((x-mu)/beta).^g).*((x-mu)/beta).^(alpha*g-1))/(beta*gamma(alpha)));

   Mat Y = Mat(X.rows, X.cols, CV_64F, Scalar(0));
   Mat temp;
   X = X - mu;
   X = X / beta;
   X.copyTo(temp);

   pow(X, g, X);
   X = -X;
   exp(X, X);
   X *= g;
   pow(temp, alpha*g-1, temp);

   multiply(X, temp, Y);
   Y *= sigma;
   Y /= (beta*tgamma(alpha));

   return Y;
 }

//evaluate polynomial with coefficients p on array x
Mat Foveate_wide_fov::polyval(Mat x, double *p, int nc) {
  //use Horner's method to evaluate polynomial
  Mat y = Mat(x.rows, x.cols, CV_64F, Scalar(0));
  y = y + p[0];
  for (int i = 1; i <= nc; i++) {
    multiply(y, x, y);
    y = y + p[i];
  }
  return y;
}

//void Foveate_wide_fov::compute_image_pyramid(Mat img, int nL, Mat *pyr) {
//
//  // double filt[5] = {8.8388e-02,   3.5355e-01,   5.3033e-01,   3.5355e-01,   8.8388e-02};
//  // Mat binom_filt_v = Mat(5, 1, CV_64F, filt);
//  // Mat binom_filt_h = Mat(1, 5, CV_64F, filt);
//  Mat tmp, tmp1;
//  double ratio = 0.5;
//  img.copyTo(pyr[0]);
//
//	for (int i = 1; i < nL; i++) {
//    //use OpenCV function for Gaussian pyramid
//    //this is similar to matlabPyrTools
//    pyrDown(pyr[i-1], pyr[i], Size(round(pyr[i-1].cols*ratio), round(pyr[i-1].rows*ratio)), BORDER_DEFAULT);
//  }
//  //upsample images in each level of the pyramid to the original size
//  for (int i = 1; i < nL; i++) {
//    resize(pyr[i], pyr[i], Size(img.cols, img.rows), 0, 0, INTER_CUBIC);
//  }
//}

// //evaluate polynomial with coefficients p on array x
// static Mat polyval(Mat x, double *p, int nc) {
//   //use Horner's method to evaluate polynomial
//   Mat y = Mat(x.rows, x.cols, CV_64F, Scalar(0));
//   y = y + p[0];
//   for (int i = 1; i <= nc; i++) {
//     multiply(y, x, y);
//     y = y + p[i];
//   }
//   return y;
// }
