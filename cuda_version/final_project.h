#ifndef __FINAL_PROJECT_H__
#define __FINAL_PROJECT_H__

#include <stdint.h>

class CG {
private:

	double* A;
	uint64_t* A_col;
	uint64_t* A_row;
	double* B;
	double* X;

	double* cudaA;
	uint64_t* cudaA_col;
	uint64_t* cudaA_row;
	double* cudaB;
	double* cudaX;

	double* cudaR;
	double* cudaR_prev;
	double* cudaP;
	double* cudaS;

	double* cudaRs;
	double* cudaAlpha;
	double* cudaBeta;

	uint64_t iter;


	void build_matrix0();

	inline double dot_product(double* a, double* b, uint64_t len);

public:

	CG();

	~CG();

	void init_host();

	void init_dev();

	int run();

	bool check();
};

#endif //__FINAL_PROJECT_H__