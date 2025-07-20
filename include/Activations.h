#pragma once
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class Activations
{
public:
	static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& X)
	{
		auto sigmoid_func = [](double x) { return 1 / (1 + std::exp(-x)); };
		return X.unaryExpr(sigmoid_func);
	}

	static Eigen::MatrixXd relu(const Eigen::MatrixXd& X)
	{
		auto relu_func = [](double x) { return x >= 0 ? x : 0; };
		return X.unaryExpr(relu_func);
	}
};
