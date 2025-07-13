#pragma once
class McCullochPittsUnit
{
	int m_x1, m_x2;

public:
	McCullochPittsUnit(int x1, int x2) : m_x1(x1), m_x2(x2) {}

	int g()
	{
		return m_x1 + m_x2;
	}
	
	int f(int threshold)
	{
		int sum = g();
		return sum >= threshold ? 1 : 0;
	}
};