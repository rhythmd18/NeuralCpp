#pragma once
#include <cmath>
#include "Layer.h"
#include <Eigen/Dense>

class Tanh : public Layer
{
	Eigen::MatrixXd A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) override;
	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override;
};