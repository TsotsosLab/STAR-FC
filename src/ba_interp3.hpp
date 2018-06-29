// Fast nearest, bi-linear and bi-cubic interpolation for 3d image data on a regular grid.
//
// Usage:
// ------
//     R = ba_interp3(F, X, Y, Z, [method])
//     R = ba_interp3(Fx, Fy, Fz, F, X, Y, Z, [method])
//
// where method is one off nearest, linear, or cubic.
//
// Fx, Fy, Fz
//         are the coordinate system in which F is given. Only the first and
//         last entry in Fx, Fy, Fz are used, and it is assumed that the
//         inbetween values are linearly interpolated.
// F       is a WxHxDxC Image with an arbitray number of channels C.
// X, Y, Z are I_1 x ... x I_n matrices with the x and y coordinates to
//         interpolate.
// R       is a I_1 x ... x I_n x C matrix, which contains the interpolated image channels.
//
// Notes:
// ------
// This method handles the border by repeating the closest values to the point accessed.
// This is different from matlabs border handling.
//
// Example
// ------
//
//    %% Interpolation of 3D volumes (e.g. distance transforms)
//    clear
//    sz=5;
//
//    % Dist
//    dist1.D = randn(sz,sz,sz);
//    [dist1.x dist1.y dist.z] = meshgrid(linspace(-1,1,sz), linspace(-1,1,sz), linspace(-1,1,sz));
//
//    R = [cos(pi/4) sin(pi/4); -sin(pi/4) cos(pi/4)];
//    RD = R * [Dx(:)'; Dy(:)'] + 250;
//    RDx = reshape(RD(1,:), size(Dx));
//    RDy = reshape(RD(2,:), size(Dy));
//
//    methods = {'nearest', 'linear', 'cubic'};
//    la=nan(1,3);
//    for i=1:3
//      la(i) = subplot(2,2,i);
//      tic;
//      IMG_R = ba_interp2(IMG, RDx, RDy, methods{i});
//      elapsed=toc;
//      imshow(IMG_R);
//      title(sprintf('Rotation and zoom using %s interpolation took %gs', methods{i}, elapsed));
//    end
//    linkaxes(la);
//
// Licence:
// --------
// GPL
// (c) 2008 Brian Amberg
// http://www.brian-amberg.de/

// modified by Yulia Kotseruba 07.04.2016
// - removed matlab headers and mex function references

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;


inline
static
int access(int M, int N, int O, int x, int y, int z) {
  if (x<0) x=0; else if (x>=M) x=M-1;
  if (y<0) y=0; else if (y>=N) y=N-1;
  if (z<0) z=0; else if (z>=O) z=O-1;
  //return y + M*(x + N*z);
  return z*N*M + x*N + y;
}

inline
static
int access_unchecked(int M, int N, int O, int x, int y, int z) {
  //return y + M*(x + N*z);
  return z*N*M + x*N + y;
}

inline
static
void indices_linear(
    int &f000_i,
    int &f100_i,
    int &f010_i,
    int &f110_i,
    int &f001_i,
    int &f101_i,
    int &f011_i,
    int &f111_i,
    const int x, const int y, const int z,
    const size_t &M, const size_t &N, const size_t &O) {
  if (x<=1 || y<=1 || z<=1 || x>=M-2 || y>=N-2 || z>=O-2) {
    f000_i = access(M,N,O, x,   y  , z);
    f100_i = access(M,N,O, x+1, y  , z);

    f010_i = access(M,N,O, x,   y+1, z);
    f110_i = access(M,N,O, x+1, y+1, z);

    f001_i = access(M,N,O, x,   y  , z+1);
    f101_i = access(M,N,O, x+1, y  , z+1);

    f011_i = access(M,N,O, x,   y+1, z+1);
    f111_i = access(M,N,O, x+1, y+1, z+1);
  } else {
    f000_i = access_unchecked(M,N,O, x,   y  , z);
    f100_i = access_unchecked(M,N,O, x+1, y  , z);

    f010_i = access_unchecked(M,N,O, x,   y+1, z);
    f110_i = access_unchecked(M,N,O, x+1, y+1, z);

    f001_i = access_unchecked(M,N,O, x,   y  , z+1);
    f101_i = access_unchecked(M,N,O, x+1, y  , z+1);

    f011_i = access_unchecked(M,N,O, x,   y+1, z+1);
    f111_i = access_unchecked(M,N,O, x+1, y+1, z+1);
  }
}

inline
static
void indices_cubic(
    int f_i[64],
    const int x, const int y, const int z,
    const size_t &M, const size_t &N, const size_t &O) {
  if (x<=2 || y<=2 || z<=2 || x>=M-3 || y>=N-3 || z>=O-3) {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          //f_i[i+4*(j+4*k)] = access(M,N,O, x+i-1, y+j-1, z+k-1);
            f_i[k*16 + i*4 + j] = access(M,N,O, x+i-1, y+j-1, z+k-1);

  } else {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          //f_i[i+4*(j+4*k)] = access_unchecked(M,N,O, x+i-1, y+j-1, z+k-1);
  			f_i[k*16 + i*4 + j] = access_unchecked(M,N,O, x+i-1, y+j-1, z+k-1);

  }
}


static
void interpolate_nearest(double *pO, const double *pF,
  const double *pX, const double *pY, const double *pZ,
  const size_t ND, const size_t M, const size_t N, const size_t O, const size_t P,
  const double s_x, const double o_x,
  const double s_y, const double o_y,
  const double s_z, const double o_z) {
  const size_t LO = M*N*O;
  for (size_t i=0; i<ND; ++i) {
    const double &x = pX[i];
    const double &y = pY[i];
    const double &z = pZ[i];

    const int x_round = int(round(s_x*x+o_x))-1;
    const int y_round = int(round(s_y*y+o_y))-1;
    const int z_round = int(round(s_z*z+o_z))-1;

    const int f00_i = access(M,N,O, x_round,y_round,z_round);
    for (size_t j=0; j<P; ++j) {
      pO[i + j*ND] = pF[f00_i + j*LO];
    }
  }
}

template <size_t P>
static
void interpolate_nearest_unrolled(double *pO, const double *pF,
  const double *pX, const double *pY, const double *pZ,
  const size_t ND, const size_t M, const size_t N, const size_t O,
  const double s_x, const double o_x,
  const double s_y, const double o_y,
  const double s_z, const double o_z) {
  const size_t LO = M*N*O;
  for (size_t i=0; i<ND; ++i) {
    const double &x = pX[i];
    const double &y = pY[i];
    const double &z = pZ[i];

    const int x_round = int(round(s_x*x+o_x))-1;
    const int y_round = int(round(s_y*y+o_y))-1;
    const int z_round = int(round(s_z*z+o_z))-1;

    const int f00_i = access(M,N,O, x_round,y_round,z_round);
    for (size_t j=0; j<P; ++j) {
      pO[i + j*ND] = pF[f00_i + j*LO];
    }
  }
}

static
void interpolate_linear(double *pO, const double *pF,
  const double *pX, const double *pY, const double *pZ,
  const size_t ND, const size_t M, const size_t N, const size_t O, const size_t P,
  const double s_x, const double o_x,
  const double s_y, const double o_y,
  const double s_z, const double o_z) {
  const size_t LO = M*N*O;
  for (size_t i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];

    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;

    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);

    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;

    const double wx0 = 1.0-dx;
    const double wx1 = dx;

    const double wy0 = 1.0-dy;
    const double wy1 = dy;

    const double wz0 = 1.0-dz;
    const double wz1 = dz;

    int f000_i, f100_i, f010_i, f110_i;
    int f001_i, f101_i, f011_i, f111_i;

    // TODO: Use openmp
    indices_linear(
        f000_i, f100_i, f010_i, f110_i,
        f001_i, f101_i, f011_i, f111_i,
        int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);

    for (size_t j=0; j<P; ++j) {

      pO[i + j*ND] =
        wz0*(
            wy0*(wx0 * pF[f000_i + j*LO] + wx1 * pF[f100_i + j*LO]) +
            wy1*(wx0 * pF[f010_i + j*LO] + wx1 * pF[f110_i + j*LO])
            )+
        wz1*(
            wy0*(wx0 * pF[f001_i + j*LO] + wx1 * pF[f101_i + j*LO]) +
            wy1*(wx0 * pF[f011_i + j*LO] + wx1 * pF[f111_i + j*LO])
            );
    }

  }
}

template <size_t P>
static
void interpolate_linear_unrolled(double *pO, const double *pF,
  const double *pX, const double *pY, const double *pZ,
  const size_t ND, const size_t M, const size_t N, const size_t O,
  const double s_x, const double o_x,
  const double s_y, const double o_y,
  const double s_z, const double o_z) {
  const size_t LO = M*N*O;
  for (size_t i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];

    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;

    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);

    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;

    const double wx0 = 1.0-dx;
    const double wx1 = dx;

    const double wy0 = 1.0-dy;
    const double wy1 = dy;

    const double wz0 = 1.0-dz;
    const double wz1 = dz;

    int f000_i, f100_i, f010_i, f110_i;
    int f001_i, f101_i, f011_i, f111_i;

    // TODO: Use openmp

    indices_linear(
        f000_i, f100_i, f010_i, f110_i,
        f001_i, f101_i, f011_i, f111_i,
        int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);

    for (size_t j=0; j<P; ++j) {

      pO[i + j*ND] =
        wz0*(
            wy0*(wx0 * pF[f000_i + j*LO] + wx1 * pF[f100_i + j*LO]) +
            wy1*(wx0 * pF[f010_i + j*LO] + wx1 * pF[f110_i + j*LO])
            )+
        wz1*(
            wy0*(wx0 * pF[f001_i + j*LO] + wx1 * pF[f101_i + j*LO]) +
            wy1*(wx0 * pF[f011_i + j*LO] + wx1 * pF[f111_i + j*LO])
            );
    }

  }
}

static
void interpolate_bicubic(double *pO, const double *pF,
  const double *pX, const double *pY, const double *pZ,
  const size_t ND, const size_t M, const size_t N, const size_t O, const size_t P,
  const double s_x, const double o_x,
  const double s_y, const double o_y,
  const double s_z, const double o_z) {
  const size_t LO = M*N*O;
  for (size_t i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];

    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;

    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);


    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;

    const double dxx = dx*dx;
    const double dxxx = dxx*dx;

    const double dyy = dy*dy;
    const double dyyy = dyy*dy;

    const double dzz = dz*dz;
    const double dzzz = dzz*dz;

    const double wx0 = 0.5 * (    - dx + 2.0*dxx -       dxxx);
    const double wx1 = 0.5 * (2.0      - 5.0*dxx + 3.0 * dxxx);
    const double wx2 = 0.5 * (      dx + 4.0*dxx - 3.0 * dxxx);
    const double wx3 = 0.5 * (         -     dxx +       dxxx);

    const double wy0 = 0.5 * (    - dy + 2.0*dyy -       dyyy);
    const double wy1 = 0.5 * (2.0      - 5.0*dyy + 3.0 * dyyy);
    const double wy2 = 0.5 * (      dy + 4.0*dyy - 3.0 * dyyy);
    const double wy3 = 0.5 * (         -     dyy +       dyyy);

    const double wz0 = 0.5 * (    - dz + 2.0*dzz -       dzzz);
    const double wz1 = 0.5 * (2.0      - 5.0*dzz + 3.0 * dzzz);
    const double wz2 = 0.5 * (      dz + 4.0*dzz - 3.0 * dzzz);
    const double wz3 = 0.5 * (         -     dzz +       dzzz);

    int f_i[64];

    indices_cubic(f_i, int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);

    for (size_t j=0; j<P; ++j) {

      pO[i + j*ND] =
		wz0*(
			wy0*(wx0 * pF[f_i[0] + j*LO] + wx1 * pF[f_i[1] + j*LO] +  wx2 * pF[f_i[2] + j*LO] + wx3 * pF[f_i[3] + j*LO]) +
			wy1*(wx0 * pF[f_i[4] + j*LO] + wx1 * pF[f_i[5] + j*LO] +  wx2 * pF[f_i[6] + j*LO] + wx3 * pF[f_i[7] + j*LO]) +
			wy2*(wx0 * pF[f_i[8] + j*LO] + wx1 * pF[f_i[9] + j*LO] +  wx2 * pF[f_i[10] + j*LO] + wx3 * pF[f_i[11] + j*LO]) +
			wy3*(wx0 * pF[f_i[12] + j*LO] + wx1 * pF[f_i[13] + j*LO] +  wx2 * pF[f_i[14] + j*LO] + wx3 * pF[f_i[15] + j*LO])
			) +
		wz1*(
			wy0*(wx0 * pF[f_i[16] + j*LO] + wx1 * pF[f_i[17] + j*LO] +  wx2 * pF[f_i[18] + j*LO] + wx3 * pF[f_i[19] + j*LO]) +
			wy1*(wx0 * pF[f_i[20] + j*LO] + wx1 * pF[f_i[21] + j*LO] +  wx2 * pF[f_i[22] + j*LO] + wx3 * pF[f_i[23] + j*LO]) +
			wy2*(wx0 * pF[f_i[24] + j*LO] + wx1 * pF[f_i[25] + j*LO] +  wx2 * pF[f_i[26] + j*LO] + wx3 * pF[f_i[27] + j*LO]) +
			wy3*(wx0 * pF[f_i[28] + j*LO] + wx1 * pF[f_i[29] + j*LO] +  wx2 * pF[f_i[30] + j*LO] + wx3 * pF[f_i[31] + j*LO])
			) +
		wz2*(
			wy0*(wx0 * pF[f_i[32] + j*LO] + wx1 * pF[f_i[33] + j*LO] +  wx2 * pF[f_i[34] + j*LO] + wx3 * pF[f_i[35] + j*LO]) +
			wy1*(wx0 * pF[f_i[36] + j*LO] + wx1 * pF[f_i[37] + j*LO] +  wx2 * pF[f_i[38] + j*LO] + wx3 * pF[f_i[39] + j*LO]) +
			wy2*(wx0 * pF[f_i[40] + j*LO] + wx1 * pF[f_i[41] + j*LO] +  wx2 * pF[f_i[42] + j*LO] + wx3 * pF[f_i[43] + j*LO]) +
			wy3*(wx0 * pF[f_i[44] + j*LO] + wx1 * pF[f_i[45] + j*LO] +  wx2 * pF[f_i[46] + j*LO] + wx3 * pF[f_i[47] + j*LO])
			) +
		wz3*(
			wy0*(wx0 * pF[f_i[48] + j*LO] + wx1 * pF[f_i[49] + j*LO] +  wx2 * pF[f_i[50] + j*LO] + wx3 * pF[f_i[51] + j*LO]) +
			wy1*(wx0 * pF[f_i[52] + j*LO] + wx1 * pF[f_i[53] + j*LO] +  wx2 * pF[f_i[54] + j*LO] + wx3 * pF[f_i[55] + j*LO]) +
			wy2*(wx0 * pF[f_i[56] + j*LO] + wx1 * pF[f_i[57] + j*LO] +  wx2 * pF[f_i[58] + j*LO] + wx3 * pF[f_i[59] + j*LO]) +
			wy3*(wx0 * pF[f_i[60] + j*LO] + wx1 * pF[f_i[61] + j*LO] +  wx2 * pF[f_i[62] + j*LO] + wx3 * pF[f_i[63] + j*LO])
			);

    }

  }
}

template <size_t P>
static
void interpolate_bicubic_unrolled(double *pO, const double *pF,
  const double *pX, const double *pY, const double *pZ,
  const size_t ND, const size_t M, const size_t N, const size_t O,
  const double s_x, const double o_x,
  const double s_y, const double o_y,
  const double s_z, const double o_z) {
  const size_t LO = M*N*O;

#pragma omp parallel for
  for (size_t i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];

    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;

    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);


    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;

    const double dxx = dx*dx;
    const double dxxx = dxx*dx;

    const double dyy = dy*dy;
    const double dyyy = dyy*dy;

    const double dzz = dz*dz;
    const double dzzz = dzz*dz;

    const double wx0 = 0.5 * (    - dx + 2.0*dxx -       dxxx);
    const double wx1 = 0.5 * (2.0      - 5.0*dxx + 3.0 * dxxx);
    const double wx2 = 0.5 * (      dx + 4.0*dxx - 3.0 * dxxx);
    const double wx3 = 0.5 * (         -     dxx +       dxxx);

    const double wy0 = 0.5 * (    - dy + 2.0*dyy -       dyyy);
    const double wy1 = 0.5 * (2.0      - 5.0*dyy + 3.0 * dyyy);
    const double wy2 = 0.5 * (      dy + 4.0*dyy - 3.0 * dyyy);
    const double wy3 = 0.5 * (         -     dyy +       dyyy);

    const double wz0 = 0.5 * (    - dz + 2.0*dzz -       dzzz);
    const double wz1 = 0.5 * (2.0      - 5.0*dzz + 3.0 * dzzz);
    const double wz2 = 0.5 * (      dz + 4.0*dzz - 3.0 * dzzz);
    const double wz3 = 0.5 * (         -     dzz +       dzzz);

    int f_i[64];

    indices_cubic(f_i, int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);

    for (size_t j=0; j<P; ++j) {
    int jLO = j*LO;
	pO[i + j*ND] =
	wz0*(
		wy0*(wx0 * pF[f_i[0] + jLO] + wx1 * pF[f_i[1] + jLO] +  wx2 * pF[f_i[2] + jLO] + wx3 * pF[f_i[3] + jLO]) +
		wy1*(wx0 * pF[f_i[4] + jLO] + wx1 * pF[f_i[5] + jLO] +  wx2 * pF[f_i[6] + jLO] + wx3 * pF[f_i[7] + jLO]) +
		wy2*(wx0 * pF[f_i[8] + jLO] + wx1 * pF[f_i[9] + jLO] +  wx2 * pF[f_i[10] + jLO] + wx3 * pF[f_i[11] + jLO]) +
		wy3*(wx0 * pF[f_i[12] + jLO] + wx1 * pF[f_i[13] + jLO] +  wx2 * pF[f_i[14] + jLO] + wx3 * pF[f_i[15] + jLO])
		) +
	wz1*(
		wy0*(wx0 * pF[f_i[16] + jLO] + wx1 * pF[f_i[17] + jLO] +  wx2 * pF[f_i[18] + jLO] + wx3 * pF[f_i[19] + jLO]) +
		wy1*(wx0 * pF[f_i[20] + jLO] + wx1 * pF[f_i[21] + jLO] +  wx2 * pF[f_i[22] + jLO] + wx3 * pF[f_i[23] + jLO]) +
		wy2*(wx0 * pF[f_i[24] + jLO] + wx1 * pF[f_i[25] + jLO] +  wx2 * pF[f_i[26] + jLO] + wx3 * pF[f_i[27] + jLO]) +
		wy3*(wx0 * pF[f_i[28] + jLO] + wx1 * pF[f_i[29] + jLO] +  wx2 * pF[f_i[30] + jLO] + wx3 * pF[f_i[31] + jLO])
		) +
	wz2*(
		wy0*(wx0 * pF[f_i[32] + jLO] + wx1 * pF[f_i[33] + jLO] +  wx2 * pF[f_i[34] + jLO] + wx3 * pF[f_i[35] + jLO]) +
		wy1*(wx0 * pF[f_i[36] + jLO] + wx1 * pF[f_i[37] + jLO] +  wx2 * pF[f_i[38] + jLO] + wx3 * pF[f_i[39] + jLO]) +
		wy2*(wx0 * pF[f_i[40] + jLO] + wx1 * pF[f_i[41] + jLO] +  wx2 * pF[f_i[42] + jLO] + wx3 * pF[f_i[43] + jLO]) +
		wy3*(wx0 * pF[f_i[44] + jLO] + wx1 * pF[f_i[45] + jLO] +  wx2 * pF[f_i[46] + jLO] + wx3 * pF[f_i[47] + jLO])
		) +
	wz3*(
		wy0*(wx0 * pF[f_i[48] + jLO] + wx1 * pF[f_i[49] + jLO] +  wx2 * pF[f_i[50] + jLO] + wx3 * pF[f_i[51] + jLO]) +
		wy1*(wx0 * pF[f_i[52] + jLO] + wx1 * pF[f_i[53] + jLO] +  wx2 * pF[f_i[54] + jLO] + wx3 * pF[f_i[55] + jLO]) +
		wy2*(wx0 * pF[f_i[56] + jLO] + wx1 * pF[f_i[57] + jLO] +  wx2 * pF[f_i[58] + jLO] + wx3 * pF[f_i[59] + jLO]) +
		wy3*(wx0 * pF[f_i[60] + jLO] + wx1 * pF[f_i[61] + jLO] +  wx2 * pF[f_i[62] + jLO] + wx3 * pF[f_i[63] + jLO])
		);

    }

  }
}


/**
 * pF - pyramid image
 * pX - x indices
 * pY - y indices
 * pZ - z indices (what layer of the pyramid should be at x,y point)
 * p0 - output image
 * h - height of the image
 * w - width of the image
 * nL - number of levels in the pyramid
 * method - one of "nearest", "linear" or "cubic"
 */
static void ba_interp3(double *pF, double *pX, double *pY, double *pZ, double *&pO, int h, int w, int nL, const char *method) {

  int M = h;
  int N = w;
  int O = nL;

  int P = 1;
  int outDims[2] = {h, w};

  int ND = h*w;

  // const double *pF = mxGetPr(F);
  // const double *pX = mxGetPr(X);
  // const double *pY = mxGetPr(Y);
  // const double *pZ = mxGetPr(Z);
  // double       *pO = mxGetPr(plhs[0]);


  if (strcmp(method, "nearest") == 0) {
      switch (P) {
        case 1: interpolate_nearest_unrolled<1>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 2: interpolate_nearest_unrolled<2>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 3: interpolate_nearest_unrolled<3>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 4: interpolate_nearest_unrolled<4>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 5: interpolate_nearest_unrolled<5>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 6: interpolate_nearest_unrolled<6>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 7: interpolate_nearest_unrolled<7>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 8: interpolate_nearest_unrolled<8>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 9: interpolate_nearest_unrolled<9>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        default:
                interpolate_nearest(pO, pF, pX, pY, pZ, ND, M, N, O, P, double(1), double(0), double(1), double(0), double(1), double(0));
                break;
      }
    } else if (strcmp(method, "linear") == 0) {
      switch (P) {
        case 1: interpolate_linear_unrolled<1>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 2: interpolate_linear_unrolled<2>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 3: interpolate_linear_unrolled<3>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 4: interpolate_linear_unrolled<4>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 5: interpolate_linear_unrolled<5>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 6: interpolate_linear_unrolled<6>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 7: interpolate_linear_unrolled<7>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 8: interpolate_linear_unrolled<8>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 9: interpolate_linear_unrolled<9>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        default:
                interpolate_linear(pO, pF, pX, pY, pZ, ND, M, N, O, P, double(1), double(0), double(1), double(0), double(1), double(0));
                break;
        }
    } else if (strcmp(method, "cubic") == 0) {
      switch (P) {
        case 1: interpolate_bicubic_unrolled<1>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 2: interpolate_bicubic_unrolled<2>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 3: interpolate_bicubic_unrolled<3>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 4: interpolate_bicubic_unrolled<4>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 5: interpolate_bicubic_unrolled<5>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 6: interpolate_bicubic_unrolled<6>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 7: interpolate_bicubic_unrolled<7>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 8: interpolate_bicubic_unrolled<8>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        case 9: interpolate_bicubic_unrolled<9>(pO, pF, pX, pY, pZ, ND, M, N, O, double(1), double(0), double(1), double(0), double(1), double(0)); break;
        default:
                interpolate_bicubic(pO, pF, pX, pY, pZ, ND, M, N, O, P, double(1), double(0), double(1), double(0), double(1), double(0));
                break;
      }
//      for (int i = 0; i < 25; i++) {
//    	  cout << pO[i] << " ";
//      }
//      cout << endl;

    } else {
        printf("Unimplemented interpolation method.\n");
    }

  }
