#pragma once
#include <Eigen/Dense>

class McCullochPittsUnit
{
	Eigen::MatrixXi m_X;
public:
	McCullochPittsUnit(Eigen::MatrixXi& X) : m_X(X) {}
	Eigen::VectorXi g()
	{
		return m_X.rowwise().sum();
	}
	Eigen::VectorXi f(Eigen::VectorXi g, int threshold)
	{
		auto f = [threshold](int x) { return x >= threshold ? 1 : 0; };
		return g.unaryExpr(f);
	}
};