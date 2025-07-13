#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Gate.h"

int main()
{
	Eigen::MatrixXi tt(4, 2);
	tt << 0, 0,
		0, 1,
		1, 0,
		1, 1;
	
	Gate gate(tt);

	std::cout << "Truth Table:\n" << tt << "\n" << std::endl;
	std::cout << "OR Gate Output:\n" << gate.OR() << "\n" << std::endl;
	std::cout << "AND Gate Output:\n" << gate.AND() << "\n" << std::endl;
	std::cout << "NOR Gate Output:\n" << gate.NOR() << "\n" << std::endl;
	std::cout << "NAND Gate Output:\n" << gate.NAND() << "\n" << std::endl;

	return 0;
}