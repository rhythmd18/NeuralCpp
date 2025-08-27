#pragma once
#include <Eigen/Dense>

class Layer
{
public:
	virtual ~Layer() = default;
	virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X) = 0;
	virtual Eigen::MatrixXd _backward(const Eigen::MatrixXd& dA) = 0;
};