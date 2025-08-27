#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class Softmax : public Layer
{
	Eigen::MatrixXd A;

public:
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) override;
	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) override;
};