#include <iostream>
#include <Eigen/Dense>
#include "include/nn/layers/Layers.h"

int main()
{
	Eigen::MatrixXd X(4, 2);
	X << 0, 0,
		0, 1,
		1, 0,
		1, 1;

	Linear layer(2, 5);
	Eigen::MatrixXd Z = layer(X);

	Sigmoid g;
	Eigen::MatrixXd A = g(Z);

	std::cout << A << std::endl;

	std::cin.get();
 	return 0;
}