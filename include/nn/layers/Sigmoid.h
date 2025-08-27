#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class Sigmoid : public Layer
{
	Eigen::MatrixXd A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) override
	{
		A = 1.0 / (1.0 + (-X.array()).exp());
		return A;
	}

	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override
	{
		return dA.array() * A.array() * (1.0 - A.array());
	}
};