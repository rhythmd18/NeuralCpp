#include "../../../include/nn/layers/Softmax.h"
#include <Eigen/Dense>

Eigen::MatrixXd Softmax::operator()(const Eigen::MatrixXd& X)
{
	A = X.array().exp().array().colwise() /
		X.array().exp().rowwise().sum().array();
	return A;
}

Eigen::MatrixXd Softmax::_backward(const Eigen::MatrixXd& dA)
{
	return A.array() * (dA.array() - (dA.array() * A.array()).rowwise().sum().array());
}