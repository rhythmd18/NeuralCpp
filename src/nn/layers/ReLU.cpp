#include "../../../include/nn/layers/ReLU.h"
#include <Eigen/Dense>

Eigen::MatrixXd ReLU::operator()(const Eigen::MatrixXd& X_)
{
	X = X_;
	A = X.cwiseMax(0.0);
	return A;
}

Eigen::MatrixXd ReLU::_backward(const Eigen::MatrixXd& dA)
{
	Eigen::MatrixXd mask = (X.array() >= 0).cast<double>();
	return dA.array() * mask.array();
}