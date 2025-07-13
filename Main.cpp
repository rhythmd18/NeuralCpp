#include <iostream>
#include <vector>
#include "Gate.h"

int main()
{
	std::vector<int> X = { 1, 1 };
	Gate gate1(X);

	std::cout << "OR: " << gate1.OR() << std::endl;
	std::cout << "AND: " << gate1.AND() << std::endl;
	std::cout << "NOR: " << gate1.NOR() << std::endl;
	std::cout << "NAND: " << gate1.NAND() << std::endl;
}