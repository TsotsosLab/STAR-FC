#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math_constants.h>
#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"ERROR: \"%s\" in %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ inline int access_(int M, int N, int O, int x, int y, int z) {
  if (x<0) x=0; else if (x>=M) x=M-1;
  if (y<0) y=0; else if (y>=N) y=N-1;
  if (z<0) z=0; else if (z>=O) z=O-1;
  //return y + M*(x + N*z);
  return z*M*N + x*N + y;
}

__device__ int access_unchecked_(int M, int N, int O, int x, int y, int z) {
  //return y + M*(x + N*z);
  return z*M*N + x*N + y;
}

__device__ inline void indices_cubic_(
    int f_i[64],
    const int x, const int y, const int z,
    const size_t &M, const size_t &N, const size_t &O) {
  if (x<=2 || y<=2 || z<=2 || x>=N-3 || y>=M-3 || z>=O-3) {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          //f_i[i+4*(j+4*k)] = access_(M,N,O, x+i-1, y+j-1, z+k-1);
          f_i[k*16 + i*4 + j] = access_(M,N,O, x+i-1, y+j-1, z+k-1);
  } else {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          //f_i[i+4*(j+4*k)] = access_unchecked_(M,N,O, x+i-1, y+j-1, z+k-1);
  			f_i[k*16 + i*4 + j] = access_unchecked_(M,N,O, x+i-1, y+j-1, z+k-1);
  }
}

__global__ void interpolate_bicubic_GPU(float *pO, const float *pF,
	const float *pZ, const size_t ND, const size_t M, const size_t N, const size_t O) {

  	int blockId = blockIdx.x*gridDim.y + blockIdx.y;
	//int threadId = threadIdx.x + blockDim.z*(threadIdx.y+blockDim.z*threadIdx.z);
	int threadId = threadIdx.z*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;

	//printf("blockID=(%i %i) threadIdx=(%i %i %i) threadId=%i\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z, threadId);

    //const float x = pX[blockId];
    //const float y = pY[blockId];
    const float x = blockIdx.x;
    const float y = blockIdx.y;
    const float z = pZ[blockId];

    const float x_floor = floor(x);
    const float y_floor = floor(y);
    const float z_floor = floor(z);


    const float dx = x-x_floor;
    const float dy = y-y_floor;
    const float dz = z-z_floor;

    const float dxx = dx*dx;
    const float dxxx = dxx*dx;

    const float dyy = dy*dy;
    const float dyyy = dyy*dy;

    const float dzz = dz*dz;
    const float dzzz = dzz*dz;


    const float wx0 = 0.5f * (    - dx + 2.0f*dxx -       dxxx);
    const float wx1 = 0.5f * (2.0f      - 5.0f*dxx + 3.0f * dxxx);
    const float wx2 = 0.5f * (      dx + 4.0f*dxx - 3.0f * dxxx);
    const float wx3 = 0.5f * (         -     dxx +       dxxx);

    const float wy0 = 0.5f * (    - dy + 2.0f*dyy -       dyyy);
    const float wy1 = 0.5f * (2.0f      - 5.0f*dyy + 3.0f * dyyy);
    const float wy2 = 0.5f * (      dy + 4.0f*dyy - 3.0f * dyyy);
    const float wy3 = 0.5f * (         -     dyy +       dyyy);

    const float wz0 = 0.5f * (    - dz + 2.0f*dzz -       dzzz);
    const float wz1 = 0.5f * (2.0f      - 5.0f*dzz + 3.0f * dzzz);
    const float wz2 = 0.5f * (      dz + 4.0f*dzz - 3.0f * dzzz);
    const float wz3 = 0.5f * (         -     dzz +       dzzz);

    __shared__ int f_i[64];

    //indices_cubic_(f_i, int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);
	
	int x_ = x_floor-1;
	int y_ = y_floor-1;
	int z_ = z_floor-1;
	
	  if (x_<=2 || y_<=2 || z_<=2 || x_>=N-3 || y_>=M-3 || z_>=O-3) {
	    //for (int i=0; i<4; ++i)
	    //  for (int j=0; j<4; ++j)
	    //    for (int k=0; k<4; ++k)
	    
	    //f_i[i+4*(j+4*k)] = access_(M,N,O, x+i-1, y+j-1, z+k-1);	  
	  
	  	f_i[threadId] = access_(M, N, O, x_+threadIdx.x-1, y_+threadIdx.y-1, z_+threadIdx.z-1);
	  
	  } else {
	    //for (int i=0; i<4; ++i)
	    //  for (int j=0; j<4; ++j)
	    //    for (int k=0; k<4; ++k)
	    //      f_i[i+4*(j+4*k)] = access_unchecked_(M,N,O, x+i-1, y+j-1, z+k-1);
	  	f_i[threadId] = access_unchecked_(M, N, O, x_+threadIdx.x-1, y_+threadIdx.y-1, z_+threadIdx.z-1);
	  }

	__syncthreads();
	


	if (threadId == 0) {

	pO[blockId] =
	wz0*(
		wy0*(wx0 * pF[f_i[0]] + wx1 * pF[f_i[1]] +  wx2 * pF[f_i[2]] + wx3 * pF[f_i[3]]) +
		wy1*(wx0 * pF[f_i[4]] + wx1 * pF[f_i[5]] +  wx2 * pF[f_i[6]] + wx3 * pF[f_i[7]]) +
		wy2*(wx0 * pF[f_i[8]] + wx1 * pF[f_i[9]] +  wx2 * pF[f_i[10]] + wx3 * pF[f_i[11]]) +
		wy3*(wx0 * pF[f_i[12]] + wx1 * pF[f_i[13]] +  wx2 * pF[f_i[14]] + wx3 * pF[f_i[15]])
		) +
	wz1*(
		wy0*(wx0 * pF[f_i[16]] + wx1 * pF[f_i[17]] +  wx2 * pF[f_i[18]] + wx3 * pF[f_i[19]]) +
		wy1*(wx0 * pF[f_i[20]] + wx1 * pF[f_i[21]] +  wx2 * pF[f_i[22]] + wx3 * pF[f_i[23]]) +
		wy2*(wx0 * pF[f_i[24]] + wx1 * pF[f_i[25]] +  wx2 * pF[f_i[26]] + wx3 * pF[f_i[27]]) +
		wy3*(wx0 * pF[f_i[28]] + wx1 * pF[f_i[29]] +  wx2 * pF[f_i[30]] + wx3 * pF[f_i[31]])
		) +
	wz2*(
		wy0*(wx0 * pF[f_i[32]] + wx1 * pF[f_i[33]] +  wx2 * pF[f_i[34]] + wx3 * pF[f_i[35]]) +
		wy1*(wx0 * pF[f_i[36]] + wx1 * pF[f_i[37]] +  wx2 * pF[f_i[38]] + wx3 * pF[f_i[39]]) +
		wy2*(wx0 * pF[f_i[40]] + wx1 * pF[f_i[41]] +  wx2 * pF[f_i[42]] + wx3 * pF[f_i[43]]) +
		wy3*(wx0 * pF[f_i[44]] + wx1 * pF[f_i[45]] +  wx2 * pF[f_i[46]] + wx3 * pF[f_i[47]])
		) +
	wz3*(
		wy0*(wx0 * pF[f_i[48]] + wx1 * pF[f_i[49]] +  wx2 * pF[f_i[50]] + wx3 * pF[f_i[51]]) +
		wy1*(wx0 * pF[f_i[52]] + wx1 * pF[f_i[53]] +  wx2 * pF[f_i[54]] + wx3 * pF[f_i[55]]) +
		wy2*(wx0 * pF[f_i[56]] + wx1 * pF[f_i[57]] +  wx2 * pF[f_i[58]] + wx3 * pF[f_i[59]]) +
		wy3*(wx0 * pF[f_i[60]] + wx1 * pF[f_i[61]] +  wx2 * pF[f_i[62]] + wx3 * pF[f_i[63]])
		);
	}
}


//
//  Allocates GPU memory, copies the data to GPU memory, computes interpolation and copies data back to CPU
// pF - pyramid image
// pZ - z indices (what layer of the pyramid should be at x,y point)
// p0 - output image
// h - height of the image
// w - width of the image
// nL - number of levels in the pyramid
// method - one of "nearest", "linear" or "cubic"
//
void ba_interp3_GPU(float *pF, float *pZ, float *&pO, int h, int w, int nL, const char *method) {

  cudaError_t cudaerr;

  //cudaerr =  cudaSetDevice(1);

  int M = h;
  int N = w;
  int O = nL;
  int ND = h*w;

  //float *pF_f = new float[M*N*O];
  //float *pZ_f = new float[M*N];
  //float *pO_f = new float[M*N];

  //double2float(pF, pF_f, M*N*O);
  //double2float(pZ, pZ_f, M*N);
  //double2float(pO, pO_f, M*N);

  float *pF_d;
  float *pZ_d;
  float *pO_d;

  cudaerr = cudaMalloc((void **) &pF_d, M*N*O*sizeof(float)); gpuErrchk(cudaerr);
  cudaerr = cudaMalloc((void **) &pZ_d, M*N*sizeof(float)); gpuErrchk(cudaerr);
  cudaerr = cudaMalloc((void **) &pO_d, M*N*sizeof(float)); gpuErrchk(cudaerr);


  cudaerr = cudaMemcpy(pF_d, pF, M*N*O*sizeof(float), cudaMemcpyHostToDevice); gpuErrchk(cudaerr);
  cudaerr = cudaMemcpy(pZ_d, pZ, M*N*sizeof(float), cudaMemcpyHostToDevice); gpuErrchk(cudaerr);
  //cudaerr = cudaMemcpy(pO_d, pO, M*N*sizeof(float), cudaMemcpyHostToDevice); gpuErrchk(cudaerr);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 block_size(4, 4, 4);
  dim3 num_blocks(M, N);


  if (strcmp(method, "nearest") == 0) {
 	//interpolate_nearest(pO_f, pF_f, pX_f, pY_f, pZ_f, ND, M, N, O);
  } else if (strcmp(method, "linear") == 0) {
    //interpolate_linear(pO_f, pF_f, pX_f, pY_f, pZ_f, ND, M, N, O);
  } else if (strcmp(method, "cubic") == 0) {

    cudaEventRecord(start);
    interpolate_bicubic_GPU <<< num_blocks, block_size >>> (pO_d, pF_d, pZ_d, ND, M, N, O);
	cudaEventRecord(stop);
	cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);
  } else {
    printf("Unimplemented interpolation method.\n");
  }

  //cudaerr = cudaGetLastError();
  //if (cudaerr != cudaSuccess) {
  //  printf("Kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  //}

  cudaerr = cudaMemcpy(pO, pO_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);  gpuErrchk(cudaerr);
  cudaEventSynchronize(stop);

  float msec = 0;
  cudaEventElapsedTime(&msec, start, stop);
  //printf("interpolate_bicubic took %0.3f msec\n", msec);

 //     for (int i = 0; i < 25; i++) {
 //   	  printf("%0.0f ", pO[i]);
 //     }
 //     printf("\n");

  cudaFree(pF_d);
  cudaFree(pZ_d);
  cudaFree(pO_d);

}


#define WARP_SIZE 32
#define BLOCK_SIZE (12*WARP_SIZE)

__global__ void reduce(float *dDst, const float *dSrc, uint dim, bool findMin)
{
    __shared__ float cache[BLOCK_SIZE];

    uint gix = threadIdx.x + blockDim.x*blockIdx.x;

#define tid threadIdx.x

    float acc = CUDART_NAN_F;

    while (gix < dim) {
    	if (findMin)
    		acc = fmin(acc, dSrc[gix]);
    	else
    		acc = fmax(acc, dSrc[gix]);
        gix += blockDim.x*gridDim.x;
    }

    cache[tid] = acc;

    uint active = blockDim.x >> 1;

    do {
        __syncthreads();
        if (tid < active)
        	if (findMin)
        		cache[tid] = fmin(cache[tid], cache[tid+active]);
        	else
        		cache[tid] = fmax(cache[tid], cache[tid+active]);
        active >>= 1;
    } while (active > 0);

    if (tid == 0)
        dDst[blockIdx.x] = cache[0];
}


#define CUDART_PI_F 3.141592654f

__global__ void compute_eyefreq_cones(int h, int w, int gaze_x, int gaze_y, float CTO,
										float alpha, float epsilon2, float dotpitch,
										float viewingdist, float *eyefreq_cones_d) {

		int threadX = blockIdx.x*blockDim.x + threadIdx.x;
		int threadY = blockIdx.y*blockDim.y + threadIdx.y;
		int threadId = threadX*blockDim.y*gridDim.y + threadY;


		if (threadX < h && threadY < w) {

			float ex = threadX - gaze_x;
			float ey = threadY - gaze_y;

			// eradius is the radial distance between each point and the point
			// of gaze in pixels
			float eradius = sqrt(ex*ex + ey*ey)*dotpitch;


			// calculate ec, the eccentricity from the foveal center, for each
			// point in the image.  ec is in degrees.
			float ec = 180.0f*atanf(eradius/viewingdist)/CUDART_PI_F;

			float eyefreq_cones = alpha*(ec+epsilon2);
			eyefreq_cones = epsilon2/eyefreq_cones;
			eyefreq_cones = eyefreq_cones * logf(1/CTO);
			eyefreq_cones = pow(eyefreq_cones, 0.3f);

			eyefreq_cones_d[threadId] = eyefreq_cones;

//			if (threadX < 5 && threadY < 5) {
//				printf("threadIdx=(%i %i) threadX=%i threadY=%i threadId=%i\n",
//						threadIdx.x, threadIdx.y, threadX, threadY, threadId);
//				printf("eyefreq_cones_d[%i]=%f\n", threadId, eyefreq_cones);
//			}
		}
}

__global__ void compute_pyrlevel_cones(int h, int w, float* min_array_d, float* max_array_d, int num_levels,
										float* eyefreq_cones_d, float* pyrlevel_cones) {
	int threadX = blockIdx.x*blockDim.x + threadIdx.x;
	int threadY = blockIdx.y*blockDim.y + threadIdx.y;
	int threadId = threadX*blockDim.y*gridDim.y + threadY;


	if (threadX < h && threadY < w) {

   	   float eyefreq_cones = eyefreq_cones_d[threadId];
   	   eyefreq_cones = (eyefreq_cones-min_array_d[0])/(max_array_d[0]-min_array_d[0]);

	  // pyrlevel is the fractional level of the pyramid which must be
	  // used at each pixel in order to match the foveal resolution
	  // function defined above.
	  //pyrlevel = maxfreq ./ eyefreq;
	  //divide(maxfreq, eyefreq_cones, pyrlevel_cones);

	  // constrain pyrlevel in order to conform to the levels of the
  	  // pyramid which have been computed.

	  eyefreq_cones = 1 - eyefreq_cones;

	  pyrlevel_cones[threadId] = max(0.0f, min((float)num_levels, (num_levels-1)*eyefreq_cones));
	}
}

__global__ void compute_eyefreq_rods(int h, int w, int gaze_x, int gaze_y, float *eyefreq_rods_d) {
	int threadX = blockIdx.x*blockDim.x + threadIdx.x;
	int threadY = blockIdx.y*blockDim.y + threadIdx.y;
	int threadId = threadX*blockDim.y*gridDim.y + threadY;

	float p[6] = {8.8814e-11,  -1.6852e-07,   1.1048e-04,  -3.1856e-02,   3.7501e+00,  -3.0283e+00};

	if (threadX < h && threadY < w) {

		float ex = threadX - gaze_x;
		float ey = threadY - gaze_y;

		// eradius is the radial distance between each point and the point
		// of gaze in meters
		float dist_px = sqrt(ex*ex + ey*ey);

	    //eyefreq_rods = polyval(dist_px, p, 5);
	    float eyefreq_rods = p[0];
	     for (int i = 1; i <= 5; i++) {
	    	 eyefreq_rods = eyefreq_rods * dist_px + p[i];
	     }
	    eyefreq_rods_d[threadId] = eyefreq_rods;
	}
}

__global__ void compute_pyrlevel_rods(int h, int w, float* max_array_d, int num_levels,
		float* eyefreq_rods_d, float* pyrlevel_rods) {

	int threadX = blockIdx.x*blockDim.x + threadIdx.x;
	int threadY = blockIdx.y*blockDim.y + threadIdx.y;
	int threadId = threadX*blockDim.y*gridDim.y + threadY;

	if (threadX < h && threadY < w) {
   	   float eyefreq_rods = eyefreq_rods_d[threadId];
   	   eyefreq_rods = eyefreq_rods/max_array_d[0];
   	   eyefreq_rods = 1 - eyefreq_rods;
   	   pyrlevel_rods[threadId] = max(0.0f, min((float) num_levels, num_levels*eyefreq_rods+2));
	}
}

void preprocess_GPU(int h, int w, float gaze_x, float gaze_y, int num_levels,
		double CTO, double alpha, double epsilon2, double dotpitch, double viewingdist,
		float* pyrlevel_cones, float* pyrlevel_rods, bool rods_and_cones){

	cudaError_t cudaerr;

	float *pyrlevel_cones_d;
	float *pyrlevel_rods_d;
	float *eyefreq_cones_d;
	float *min_array_d;
	float *max_array_d;
	
  	cudaerr = cudaMalloc((void **) &pyrlevel_cones_d, h*w*sizeof(float)); gpuErrchk(cudaerr);
  	cudaerr = cudaMalloc((void **) &pyrlevel_rods_d, h*w*sizeof(float)); gpuErrchk(cudaerr);
  	cudaerr = cudaMalloc((void **) &eyefreq_cones_d, h*w*sizeof(float)); gpuErrchk(cudaerr);

  	int num_threads = 32;
	dim3 block_size(num_threads, num_threads, 1);
  	dim3 num_blocks((int) ceil(h/(double)num_threads), (int) ceil(w/(double)num_threads), 1);

  	//printf("height = %i width = %i BLOCK SIZE (64, 64)   NUM BLOCKS (%f, %f) \n", h, w, ceil(h/(double)num_threads), ceil(w/(double)num_threads));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

  	cudaEventRecord(start, 0);
  	compute_eyefreq_cones <<< num_blocks, block_size >>> (h, w, gaze_x, gaze_y, CTO, alpha, epsilon2, dotpitch, viewingdist, eyefreq_cones_d);
    cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);


    int nblocks = ceil(h*w/(float)BLOCK_SIZE);
    cudaerr = cudaMalloc((void **) &min_array_d, nblocks*sizeof(float)); gpuErrchk(cudaerr);
    cudaerr = cudaMalloc((void **) &max_array_d, nblocks*sizeof(float)); gpuErrchk(cudaerr);

    reduce<<<nblocks,BLOCK_SIZE>>>(min_array_d, eyefreq_cones_d, h*w, true);
    cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

    reduce<<<1,BLOCK_SIZE>>>(min_array_d, min_array_d, nblocks, true);
    cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

    reduce<<<nblocks,BLOCK_SIZE>>>(max_array_d, eyefreq_cones_d, h*w, false);
    cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

    reduce<<<1,BLOCK_SIZE>>>(max_array_d, max_array_d, nblocks, false);
    cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

    compute_pyrlevel_cones <<< num_blocks, block_size >>> (h, w, min_array_d, max_array_d, num_levels, eyefreq_cones_d, pyrlevel_cones_d);
    cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

    cudaerr = cudaMemcpy(pyrlevel_cones, pyrlevel_cones_d, h*w*sizeof(float), cudaMemcpyDeviceToHost);  gpuErrchk(cudaerr);

    if (rods_and_cones) {
		compute_eyefreq_rods <<< num_blocks, block_size >>> (h, w, gaze_x, gaze_y, eyefreq_cones_d);
		cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

		reduce<<<nblocks,BLOCK_SIZE>>>(max_array_d, eyefreq_cones_d, h*w, false);
		cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

		reduce<<<1,BLOCK_SIZE>>>(max_array_d, max_array_d, nblocks, false);
		cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

		compute_pyrlevel_rods <<< num_blocks, block_size >>> (h, w, max_array_d, num_levels, eyefreq_cones_d, pyrlevel_rods_d);
		cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);

		cudaerr = cudaMemcpy(pyrlevel_rods, pyrlevel_rods_d, h*w*sizeof(float), cudaMemcpyDeviceToHost);  gpuErrchk(cudaerr);
		cudaerr = cudaGetLastError(); gpuErrchk(cudaerr);
    }

    cudaEventRecord(stop, 0);

    cudaFree(pyrlevel_cones_d);
    cudaFree(pyrlevel_rods_d);
    cudaFree(eyefreq_cones_d);
    cudaFree(min_array_d);
    cudaFree(max_array_d);
}
