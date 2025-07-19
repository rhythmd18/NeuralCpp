#include <iostream>
#include <Eigen/Dense>
#include "Perceptron.h"

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

	Eigen::VectorXi Y(4);
	Y << 0, 1, 1, 1;

	Perceptron perceptron(X);
	perceptron.Train(Y, 100);

	std::cout << "Prediction: \n" << perceptron.Predict() << std::endl;

	return 0;
}