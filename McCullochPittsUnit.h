#pragma once
#include <vector>

class McCullochPittsUnit
{
	std::vector<int> m_X;

public:
	McCullochPittsUnit(std::vector<int>& X) : m_X(X) {}
	int g()
	{
		int sum = 0;
		for (int x : m_X) sum += x;
		return sum;
	}
	int f(int threshold)
	{
		int sum = g();
		return sum >= threshold ? 1 : 0;
	}
};