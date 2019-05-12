#include <iostream>
#include <chrono> 

#include "final_project.h"

using namespace std;
using namespace std::chrono;

#define CHECK

int main() {

	CG* cg = new CG();

	cg->init_host();
	cg->init_dev();

	auto start = high_resolution_clock::now(); 
	int iter = cg->run();
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start);

#ifdef CHECK
	double seq_time = cg->check();
	cout << "Speed-up: " << seq_time / (double)duration.count() << endl;
	if(seq_time != 0)
		cout << "Test Passed" << endl;
	else
		cout << "Test Failed" << endl;
	cout << endl;

#endif //CHECK

	cout << "Number of iterations: " << iter << endl;
	cout << "Execution time: " << duration.count() << " ms" << endl; 

	delete cg;
	return 0;
}
