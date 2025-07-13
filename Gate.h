#pragma once
#include <vector>
#include "McCullochPittsUnit.h"

class Gate
{
	std::vector<int> m_X;
public:
	Gate(std::vector<int>& X) : m_X(X) {}

	int OR()
	{
		int threshold = 1;
		McCullochPittsUnit unit(m_X);
		return unit.f(threshold);
	}

	int AND()
	{
		int threshold = 2;
		McCullochPittsUnit unit(m_X);
		return unit.f(threshold);
	}

	int NOR()
	{
		int result = OR();
		return result == 1 ? 0 : 1;
	}

	int NAND()
	{
		int result = AND();
		return result == 1 ? 0 : 1;
	}
};

