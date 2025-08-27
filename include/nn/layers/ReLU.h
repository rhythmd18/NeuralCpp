#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class ReLU : public Layer
{
	Eigen::MatrixXd X, A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X_) override
	{
		X = X_;
		A = X.cwiseMax(0.0);
		return A;
	}

	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override
	{
		Eigen::MatrixXd mask = (X.array() >= 0).cast<double>();
		return dA.array() * mask.array();
	}
};