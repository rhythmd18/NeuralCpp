#include "../../../include/nn/layers/Sigmoid.h"
#include <Eigen/Dense>

Eigen::MatrixXd Sigmoid::operator()(const Eigen::MatrixXd& X)
{
	A = 1.0 / (1.0 + (-X.array()).exp());
	return A;
}

Eigen::MatrixXd Sigmoid::_backward(const Eigen::MatrixXd& dA)
{
	return dA.array() * A.array() * (1.0 - A.array());
}