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
	Eigen::MatrixXd out = layer(X);

	std::cout << out << std::endl;

	std::cin.get();
 	return 0;
}