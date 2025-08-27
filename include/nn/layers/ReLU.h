#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class ReLU : public Layer
{
	Eigen::MatrixXd X, A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X_) override;
	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override;
};