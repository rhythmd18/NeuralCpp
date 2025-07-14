#include <iostream>
#include <Eigen/Dense>
#include "Perceptron.h"
#include "Gate.h"

int main()
{
	/*Eigen::MatrixXi tt(4, 2);
	tt << 0, 0,
		0, 1,
		1, 0,
		1, 1;
	
	Gate gate(tt);

	std::cout << "Truth Table:\n" << tt << "\n" << std::endl;
	std::cout << "OR Gate Output:\n" << gate.OR() << "\n" << std::endl;
	std::cout << "AND Gate Output:\n" << gate.AND() << "\n" << std::endl;
	std::cout << "NOR Gate Output:\n" << gate.NOR() << "\n" << std::endl;
	std::cout << "NAND Gate Output:\n" << gate.NAND() << "\n" << std::endl;*/

	Eigen::MatrixXd X(4, 2);
	X << 0, 0,
		0, 1,
		1, 0,
		1, 1;

	Eigen::VectorXd W(2);
	W << 1, 1; // Sample weights

	Perceptron perceptron(X, W);

	Eigen::VectorXd g = perceptron.g();
	double threshold = 0.5; // Sample threshold
	Eigen::VectorXd y = perceptron.f(g, threshold);

	std::cout << "Input: \n" << X << "\n" << std::endl;
	std::cout << "Weights: \n" << W.transpose() << "\n" << std::endl;
	std::cout << "Activation (g): \n" << g.transpose() << "\n" << std::endl;
	std::cout << "Output (y): \n" << y.transpose() << "\n" << std::endl;

	return 0;
}