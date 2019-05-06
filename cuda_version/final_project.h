#ifndef __FINAL_PROJECT_H__
#define __FINAL_PROJECT_H__

#include <stdint.h>

class CG {
private:

	float* A;
	uint64_t* A_col;
	uint64_t* A_row;
	float* B;
	float* X;

	float* cudaA;
	uint64_t* cudaA_col;
	uint64_t* cudaA_row;
	float* cudaB;
	float* cudaX;

	float* cudaR;
	float* cudaR_prev;
	float* cudaP;
	float* cudaS;

	float* cudaRs;
	float* cudaAlpha;
	float* cudaBeta;

	uint64_t iter;

public:
	
	CG();

	~CG();

	void build_matrix0();

	void init_host();

	void init_dev();

	int run();

	bool check();
};

#endif //__FINAL_PROJECT_H__