#pragma once
#include <cmath>
#include "Layer.h"
#include <Eigen/Dense>

class Tanh : public Layer
{
	Eigen::MatrixXd A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) override
	{
		A = X.array().tanh().matrix();
		return A;
	}

	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override
	{
		return dA.array() * (1.0 - dA.array().pow(2));
	}
};