void extern ba_interp3_GPU(float *pF, float *pZ, float *&pO, int h, int w, int nL, const char *method);
void extern preprocess_GPU(int h, int w, float gaze_x, float gaze_y, int num_levels,
		double CTO, double alpha, double epsilon2, double dotpitch, double viewingdist,
		float* pyrlevel_cones, float* pyrlevel_rods, bool rods_and_cones);
