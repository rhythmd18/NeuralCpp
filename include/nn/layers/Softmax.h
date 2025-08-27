#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class Softmax : public Layer
{
	Eigen::MatrixXd A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) override
	{
		A = X.array().exp().array().colwise() / 
			X.array().exp().rowwise().sum().array();
		return A;
	}

	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override
	{
		return A.array() * (dA.array() - (dA.array() * A.array()).rowwise().sum().array());
	}
};