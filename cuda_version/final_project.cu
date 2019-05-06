#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cuda_profiler_api.h>

#include "final_project.h"
#include <stdint.h>
#include <iostream>
#include <numeric>

#define N 1000
#define A_SIZE 3*N-2
#define MAX_ITER 5000
#define TOL .00001

#define THREADS 128
#define BLOCKS 4
#define TID threadIdx.x+blockDim.x*blockIdx.x
#define STRIDE blockDim.x*gridDim.x

using namespace std;

struct GlobalConstants {

	float* A;			// Compressed sparse row matrix
	uint64_t* A_col;	// Column index for entries in A
	uint64_t* A_row;	// Prefix sum of non-zero elements in each row of A
	float* B;
	float* X;

	float* P;
	float* S;

	float* Rs;
	float* Alpha;
	float* Beta;
};

__constant__ GlobalConstants cuParams;




__global__ void kernelDot(float* a, float* b, float* dest) {

	__shared__ float tmp[THREADS];

	float temp = 0;
	for(uint64_t i = TID; i < N; i += STRIDE) {
		temp += a[i] * b[i];
	}
	tmp[threadIdx.x] = temp;

	__syncthreads();
	for(uint64_t i = THREADS>>1; i != 0; i >>= 1) {
		if(threadIdx.x < i) {
			tmp[threadIdx.x] += tmp[threadIdx.x + i];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		//atomicAdd(dest, tmp[0]);		//work with float?
		*dest = tmp[0];
	}
}

__global__ void kernelMatMul() {

	for(uint64_t i = TID; i < N; i += STRIDE) {
		float temp = 0;
		for(uint64_t k = cuParams.A_row[i]; k < cuParams.A_row[i + 1]; k++) {
			temp += cuParams.A[k] * cuParams.P[cuParams.A_col[k]];
		}
		cuParams.S[i] = temp;
	}
}

__global__ void kernelScalar(float* a, float* b) {

	if(TID == 0)
		*(a) = *(b) / *(a);
}

__global__ void kernelAlpha(float* R, float* R_prev) {	//Seperate this out?

	for(uint64_t i = TID; i < N; i += STRIDE) {
		cuParams.X[i] += *(cuParams.Alpha) * cuParams.P[i];
		R[i] = R_prev[i] - *(cuParams.Alpha) * cuParams.S[i];
	}
}

__global__ void kernelBeta(float* R) {

	for(uint64_t i = TID; i < N; i += STRIDE) {
		cuParams.P[i] = R[i] - *(cuParams.Beta) * cuParams.P[i];
	}
}



CG::CG() {

	A = NULL;
	A_col = NULL;
	A_row = NULL;
	B = NULL;
	X = NULL;

	cudaA = NULL;
	cudaA_col = NULL;
	cudaA_row = NULL;
	cudaB = NULL;
	cudaX = NULL;

	cudaR = NULL;
	cudaR_prev = NULL;
	cudaP = NULL;
	cudaS = NULL;

	cudaRs = NULL;
	cudaAlpha = NULL;
	cudaBeta = NULL;
}

CG::~CG() {

	if(A) {
		delete[] A;
		delete[] A_col;
		delete[] A_row;
		delete[] B;
		delete[] X;
	}

	if(cudaA) {
		cudaFree(cudaA);
		cudaFree(cudaA_col);
		cudaFree(cudaA_row);
		cudaFree(cudaB);
		cudaFree(cudaX);

		cudaFree(cudaR);
		cudaFree(cudaR_prev);
		cudaFree(cudaP);
		cudaFree(cudaS);

		cudaFree(cudaRs);
		cudaFree(cudaAlpha);
		cudaFree(cudaBeta);
	}
}

void CG::build_matrix0() {
	/* A in non sparse matrix form
	*  x = 2, y = -1
	*	|----------------------|
	*	|x y 0 0 0 0 .... 0 0 0|
	*	|y x y 0 0 0 .... 0 0 0|
	*	|0 y x y 0 0 .... 0 0 0|
	*	|. .				. .|
	*	|. .				. .|
	*	|. .				. .|
	*	|0 0 0 0 0 0 .... 0 y x|
	*	|----------------------|
	*
	*/

	A[0] = 2;
	A_col[0] = 0;
	A[1] = -1;
	A_col[1] = 1;

	uint64_t col = 1;
	for(uint64_t i = 2; i < A_SIZE - 2; i += 3) {
		A[i] = -1;
		A[i + 1] = 2;
		A[i + 2] = -1;

		A_col[i] = col;
		A_col[i + 1] = col + 1;
		A_col[i + 2] = col + 2;
		col++;
	}

	A[A_SIZE - 2] = -1;
	A_col[A_SIZE - 2] = N - 2;
	A[A_SIZE - 1] = 2;
	A_col[A_SIZE - 1] = N - 1;

	A_row[0] = 0;
	A_row[1] = 2;

	for(uint64_t i = 2; i < N; i++) {
		A_row[i] = A_row[i - 1] + 3;
	}

	A_row[N] = A_row[N - 1] + 2;

	for(uint64_t i = 0; i < N; i++) {
		B[i] = 1;
		X[i] = 0;
	}
}

void CG::init_host() {

	A = new float[A_SIZE]();
	A_col = new uint64_t[A_SIZE]();
	A_row = new uint64_t[N + 1]();
	B = new float[N]();
	X = new float[N]();

	build_matrix0();
}

void CG::init_dev() {
	
	cudaMalloc(&cudaA, sizeof(float) * A_SIZE);
	cudaMalloc(&cudaA_col, sizeof(uint64_t) * A_SIZE);
	cudaMalloc(&cudaA_row, sizeof(uint64_t) * (N + 1));
	cudaMalloc(&cudaB, sizeof(float) * N);
	cudaMalloc(&cudaX, sizeof(float) * N);

	cudaMalloc(&cudaR, sizeof(float) * N);
	cudaMalloc(&cudaR_prev, sizeof(float) * N);
	cudaMalloc(&cudaP, sizeof(float) * N);
	cudaMalloc(&cudaS, sizeof(float) * N);

	cudaMalloc(&cudaRs, sizeof(float));
	cudaMalloc(&cudaAlpha, sizeof(float));
	cudaMalloc(&cudaBeta, sizeof(float));

	cudaMemcpy(cudaA, A, sizeof(float) * A_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaA_col, A_col, sizeof(uint64_t) * A_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaA_row, A_row, sizeof(uint64_t) * (N + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, B, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaX, X, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(cudaR, cudaB, sizeof(float) * N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(cudaR_prev, cudaB, sizeof(float) * N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(cudaP, cudaB, sizeof(float) * N, cudaMemcpyDeviceToDevice);

	GlobalConstants params;
	params.A = cudaA;
	params.A_col = cudaA_col;
	params.A_row = cudaA_row;
	params.B = cudaB;
	params.X = cudaX;

	params.P = cudaP;
	params.S = cudaS;

	params.Rs = cudaRs;
	params.Alpha = cudaAlpha;
	params.Beta = cudaBeta;
	cudaMemcpyToSymbol(cuParams, &params, sizeof(GlobalConstants));
}

int CG::run() {

	dim3 blockDim(THREADS);
	dim3 gridDim(BLOCKS);
	dim3 gridDimDot(1);				//Single block only?

	float error;
	float* swap;
	iter = 0;

	kernelDot<<<gridDimDot, blockDim>>>(cudaR, cudaR, cudaRs);
	cudaMemcpy(&error, cudaRs, sizeof(float), cudaMemcpyDeviceToHost);

	while(error > TOL && iter < MAX_ITER) {
		kernelMatMul<<<gridDim, blockDim>>>();

		swap = cudaR_prev;
		cudaR_prev = cudaR;
		cudaR = swap;

		kernelDot<<<gridDimDot, blockDim>>>(cudaR_prev, cudaR_prev, cudaBeta);
		kernelDot<<<gridDimDot, blockDim>>>(cudaP, cudaS, cudaAlpha);
		kernelScalar<<<1, 1>>>(cudaAlpha, cudaBeta);

		kernelAlpha<<<gridDim, blockDim>>>(cudaR, cudaR_prev);

		kernelDot<<<gridDimDot, blockDim>>>(cudaR, cudaR, cudaRs);
		cudaMemcpy(&error, cudaRs, sizeof(float), cudaMemcpyDeviceToHost);		//Need sync?

		kernelScalar<<<1, 1>>>(cudaBeta, cudaRs);
		kernelBeta<<<gridDim, blockDim>>>(cudaR);

		iter++;
	}

	cudaMemcpy(X, cudaX, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return iter;
}

bool CG::check() {

	int k = 0;

	float* r = new float[N]();
	float* r_prev = new float[N]();
	float* p = new float[N]();
	float* s = new float[N]();
	float* x = new float[N]();
	float* r_swap;
	float alpha, beta;

	copy(B, B + N, r);
	copy(B, B + N, r_prev);
	fill(x, x + N, 0);

	while(inner_product(r, r + N, r, 0) > TOL && k < MAX_ITER) {
		if (k == 0) {
			copy(r, r + N, p);
		}
		else {
			beta = (float)inner_product(r, r + N, r, 0) / inner_product(r_prev, r_prev + N, r_prev, 0);

			for (uint64_t i = 0; i < N; i++){
				p[i] = r[i] + beta * p[i];
			}
		}

		r_swap = r_prev;
		r_prev = r;
		r = r_swap;

		for (uint64_t row_i = 0; row_i < N; row_i++) {
			float sum = 0;

			for(uint64_t i = A_row[row_i]; i < A_row[row_i + 1]; i++) {
				sum += A[i] * p[A_col[i]];
			}

			s[row_i] = sum;
		}

		alpha = (float)inner_product(r_prev, r_prev + N, r_prev, 0) / inner_product(p, p + N, s, 0);

		for (uint64_t i = 0; i < N; i++) {
			x[i] +=  alpha * p[i];
			r[i] = r_prev[i] - alpha * s[i];
		}

		k++;
	}

	bool result = true;

	for(uint64_t i = 0; i < N; i++) {
		if(X[i] != x[i]) {
			result = false;
			break;
		}
	}

	cout << "Number of iterations seq: " << k << endl;

	delete[] r;
	delete[] r_prev;
	delete[] p;
	delete[] s;
	delete[] x;

	return result;
}
