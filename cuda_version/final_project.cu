#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cuda_profiler_api.h>

#include "final_project.h"
#include <stdint.h>
#include <iostream>
#include <chrono>

#define N 100000
#define A_SIZE 3*N-2
#define MAX_ITER 50000
#define TOL 1e-5
#define ERROR 1e-5

#define THREADS 512
#define BLOCKS (N+THREADS-1)/THREADS
#define DOT_BLOCKS 256							// Needs to be power of 2 >= BLOCKS
#define TID threadIdx.x+blockDim.x*blockIdx.x
#define STRIDE blockDim.x*gridDim.x

using namespace std;
using namespace std::chrono; 

struct GlobalConstants {

	double* A;			// Compressed sparse row matrix
	uint64_t* A_col;	// Column index for entries in A
	uint64_t* A_row;	// Prefix sum of non-zero elements in each row of A
	double* B;
	double* X;

	double* P;
	double* S;
	double* D;

	double* Rs;
	double* Alpha;
	double* Beta;
};

__constant__ GlobalConstants cuParams;




__global__ void kernelDotPartial(double* a, double* b) {

	__shared__ double tmp[THREADS];

	double temp = 0;
	for(uint64_t i = TID; i < N; i += STRIDE) {
		temp += a[i] * b[i];
	}
	tmp[threadIdx.x] = temp;

	for(uint64_t i = THREADS >> 1; i >= 1; i >>= 1) {
		__syncthreads();
		if(threadIdx.x < i) {
			tmp[threadIdx.x] += tmp[threadIdx.x + i];
		}
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		cuParams.D[blockIdx.x] = tmp[0];
	}
}

__global__ void kernelDotSum(double* a, double* b, double* dest) {

	__shared__ double tmp[DOT_BLOCKS];

	double temp = 0;
	if(threadIdx.x < BLOCKS) {
		temp = cuParams.D[threadIdx.x];
	}
	tmp[threadIdx.x] = temp;

	for(uint64_t i = DOT_BLOCKS >> 1; i >= 1; i >>= 1) {
		__syncthreads();
		if(threadIdx.x < i) {
			tmp[threadIdx.x] += tmp[threadIdx.x + i];
		}
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		*dest = tmp[0];
	}
}

__global__ void kernelMatMul() {

	for(uint64_t i = TID; i < N; i += STRIDE) {
		double temp = 0;
		for(uint64_t k = cuParams.A_row[i]; k < cuParams.A_row[i + 1]; k++) {
			temp += cuParams.A[k] * cuParams.P[cuParams.A_col[k]];
		}
		cuParams.S[i] = temp;
	}
}

__global__ void kernelScalar(double* a, double* b) {

	if(threadIdx.x == 0)
		(*a) = (*b) / (*a);
}

__global__ void kernelAlpha(double* R, double* R_prev) {

	for(uint64_t i = TID; i < N; i += STRIDE) {
		cuParams.X[i] += (*(cuParams.Alpha)) * cuParams.P[i];
		R[i] = R_prev[i] - (*(cuParams.Alpha)) * cuParams.S[i];
	}
}

__global__ void kernelBeta(double* R) {

	for(uint64_t i = TID; i < N; i += STRIDE) {
		cuParams.P[i] = R[i] + (*(cuParams.Beta)) * cuParams.P[i];
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
	cudaD = NULL;

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
		cudaFree(cudaD);

		cudaFree(cudaRs);
		cudaFree(cudaAlpha);
		cudaFree(cudaBeta);
	}
}

void CG::build_matrix0() {
	/* A in non sparse matrix form
	*
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
	double x = 2;
	double y = -1;

	A[0] = x;
	A_col[0] = 0;
	A[1] = y;
	A_col[1] = 1;

	uint64_t col = 0;
	for(uint64_t i = 2; i < A_SIZE - 2; i += 3) {
		A[i] = y;
		A[i + 1] = x;
		A[i + 2] = y;

		A_col[i] = col;
		A_col[i + 1] = col + 1;
		A_col[i + 2] = col + 2;
		col++;
	}

	A[A_SIZE - 2] = y;
	A_col[A_SIZE - 2] = N - 2;
	A[A_SIZE - 1] = x;
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

	A = new double[A_SIZE]();
	A_col = new uint64_t[A_SIZE]();
	A_row = new uint64_t[N + 1]();
	B = new double[N]();
	X = new double[N]();

	build_matrix0();
}

void CG::init_dev() {
	
	cudaMalloc(&cudaA, sizeof(double) * A_SIZE);
	cudaMalloc(&cudaA_col, sizeof(uint64_t) * A_SIZE);
	cudaMalloc(&cudaA_row, sizeof(uint64_t) * (N + 1));
	cudaMalloc(&cudaB, sizeof(double) * N);
	cudaMalloc(&cudaX, sizeof(double) * N);

	cudaMalloc(&cudaR, sizeof(double) * N);
	cudaMalloc(&cudaR_prev, sizeof(double) * N);
	cudaMalloc(&cudaP, sizeof(double) * N);
	cudaMalloc(&cudaS, sizeof(double) * N);
	cudaMalloc(&cudaD, sizeof(double) * BLOCKS);

	cudaMalloc(&cudaRs, sizeof(double));
	cudaMalloc(&cudaAlpha, sizeof(double));
	cudaMalloc(&cudaBeta, sizeof(double));

	cudaMemcpy(cudaA, A, sizeof(double) * A_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaA_col, A_col, sizeof(uint64_t) * A_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaA_row, A_row, sizeof(uint64_t) * (N + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, B, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaX, X, sizeof(double) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(cudaR, cudaB, sizeof(double) * N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(cudaR_prev, cudaB, sizeof(double) * N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(cudaP, cudaB, sizeof(double) * N, cudaMemcpyDeviceToDevice);

	GlobalConstants params;
	params.A = cudaA;
	params.A_col = cudaA_col;
	params.A_row = cudaA_row;
	params.B = cudaB;
	params.X = cudaX;

	params.P = cudaP;
	params.S = cudaS;
	params.D = cudaD;

	params.Rs = cudaRs;
	params.Alpha = cudaAlpha;
	params.Beta = cudaBeta;
	cudaMemcpyToSymbol(cuParams, &params, sizeof(GlobalConstants));
}

inline void CG::cuda_dot_product(double* a, double* b, double* dest) {

	kernelDotPartial<<<BLOCKS, THREADS>>>(a, b);
	kernelDotSum<<<1, DOT_BLOCKS>>>(a, b, dest);
}

int CG::run() {

	double error;
	double* swap;
	iter = 0;

	cuda_dot_product(cudaR, cudaR, cudaRs);
	cudaMemcpy(&error, cudaRs, sizeof(double), cudaMemcpyDeviceToHost);

	while(error > TOL && iter < MAX_ITER) {
		kernelMatMul<<<BLOCKS, THREADS>>>();

		swap = cudaR_prev;
		cudaR_prev = cudaR;
		cudaR = swap;

		cuda_dot_product(cudaR_prev, cudaR_prev, cudaBeta);
		cuda_dot_product(cudaP, cudaS, cudaAlpha);
		kernelScalar<<<1, 1>>>(cudaAlpha, cudaBeta);

		kernelAlpha<<<BLOCKS, THREADS>>>(cudaR, cudaR_prev);

		cuda_dot_product(cudaR, cudaR, cudaRs);
		cudaMemcpy(&error, cudaRs, sizeof(double), cudaMemcpyDeviceToHost);

		kernelScalar<<<1, 1>>>(cudaBeta, cudaRs);
		kernelBeta<<<BLOCKS, THREADS>>>(cudaR);

		iter++;
	}

	cudaMemcpy(X, cudaX, sizeof(double) * N, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return iter;
}

inline double CG::dot_product(double* a, double* b, uint64_t len) {

	double sum = 0;
	for(uint64_t i = 0; i < len; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

double CG::check() {

	int k = 0;

	double* r = new double[N]();
	double* r_prev = new double[N]();
	double* p = new double[N]();
	double* s = new double[N]();
	double* x = new double[N]();
	double* r_swap;
	double alpha, beta;

	copy(B, B + N, r);
	copy(B, B + N, r_prev);
	fill(x, x + N, 0);

	auto start = high_resolution_clock::now(); 

	while(dot_product(r, r, N) > TOL && k < MAX_ITER) {
		if (k == 0) {
			copy(r, r + N, p);
		}
		else {
			beta = (double)dot_product(r, r, N) / dot_product(r_prev, r_prev, N);

			for (uint64_t i = 0; i < N; i++){
				p[i] = r[i] + beta * p[i];
			}
		}

		r_swap = r_prev;
		r_prev = r;
		r = r_swap;

		for (uint64_t row_i = 0; row_i < N; row_i++) {
			double sum = 0;
			for(uint64_t i = A_row[row_i]; i < A_row[row_i + 1]; i++) {
				sum += A[i] * p[A_col[i]];
			}
			s[row_i] = sum;
		}

		alpha = (double)dot_product(r_prev, r_prev, N) / dot_product(p, s, N);

		for (uint64_t i = 0; i < N; i++) {
			x[i] +=  alpha * p[i];
			r[i] = r_prev[i] - alpha * s[i];
		}

		k++;
	}

	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start);

	bool result = true;

	for(uint64_t i = 0; i < N; i++) {
		if(X[i] > x[i] + ERROR || X[i] < x[i] - ERROR) {
			result = false;
			break;
		}
	}

	cout << "Number of iterations sequential: " << k << endl;
	cout << "Execution time sequential: " << duration.count() << " ms" << endl; 

	delete[] r;
	delete[] r_prev;
	delete[] p;
	delete[] s;
	delete[] x;

	if(result)
		return duration.count();
	return 0;
}
