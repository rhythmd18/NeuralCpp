#pragma once
#include <Eigen/Dense>
#include "McCullochPittsUnit.h"

class Gate
{
	Eigen::MatrixXi m_X;
public:
	Gate(Eigen::MatrixXi& X) : m_X(X) {}

	Eigen::VectorXi OR()
	{
		McCullochPittsUnit unit(m_X);
		Eigen::VectorXi g = unit.g();
		return unit.f(g, 1); // OR gate threshold is 1
	}

	Eigen::VectorXi AND()
	{
		McCullochPittsUnit unit(m_X);
		Eigen::VectorXi g = unit.g();
		return unit.f(g, 2); // AND gate threshold is 2
	}

	Eigen::VectorXi NOR()
	{
		McCullochPittsUnit unit(m_X);
		Eigen::VectorXi g = unit.g();
		Eigen::VectorXi f = unit.f(g, 1);
		auto func = [](int x) { return x == 0 ? 1 : 0; };
		return f.unaryExpr(func); // NOR gate is the negation of OR
	}

	Eigen::VectorXi NAND()
	{
		McCullochPittsUnit unit(m_X);
		Eigen::VectorXi g = unit.g();
		Eigen::VectorXi f = unit.f(g, 2);
		auto func = [](int x) { return x == 0 ? 1 : 0; };
		return f.unaryExpr(func); // NAND gate is the negation of AND
	}
};

