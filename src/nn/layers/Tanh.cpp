#include "../../../include/nn/layers/Tanh.h"
#include <Eigen/Dense>

Eigen::MatrixXd Tanh::operator()(const Eigen::MatrixXd& X)
{
	A = X.array().tanh().matrix();
	return A;
}

Eigen::MatrixXd Tanh::_backward(const Eigen::MatrixXd& dA)
{
	return dA.array() * (1.0 - dA.array().pow(2));
}