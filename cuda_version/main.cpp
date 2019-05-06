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
	bool correct = cg->check();
	if(correct)
		cout << "Test Passed" << endl;
	else
		cout << "Test Failed" << endl;

#endif //CHECK

	cout << "Number of iterations: " << iter << endl;
	cout << duration.count() << " ms" << endl; 

	delete cg;
	return 0;
}
