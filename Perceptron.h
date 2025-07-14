#pragma once
#include <Eigen/Dense>

class Perceptron
{
	Eigen::MatrixXd m_X;
	Eigen::VectorXd m_W;

public:
	Perceptron(Eigen::MatrixXd& X, Eigen::VectorXd& W) : m_X(X), m_W(W)
	{
		if (m_X.cols() != m_W.rows())
		{
			throw std::invalid_argument("Number of features in X must match number of weights in W.");
		}
	}
	Eigen::VectorXd g()
	{
		return m_X * m_W;
	}
	Eigen::VectorXd f(Eigen::VectorXd g, double threshold)
	{
		auto f = [threshold](double x) { return x >= threshold ? 1.0 : 0.0; };
		return g.unaryExpr(f);
	}
};

