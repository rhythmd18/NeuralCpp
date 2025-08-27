#include "../../../include/nn/criteria/BinaryCrossEntropyLoss.h"
#include <Eigen/Dense>

double BinaryCrossEntropyLoss::operator()(const Eigen::MatrixXd& y_, const Eigen::MatrixXd& out_)
{
	y = y_;
	out = out_;
	Eigen::Index m = y.rows();
	return -(y.array() * out.array().log() + (1 - y.array()) * (1 - out.array()).log()).sum() / m;
}

Eigen::MatrixXd BinaryCrossEntropyLoss::_backward()
{
	Eigen::Index m = y.rows();
	return ((1 - y.array()) / (1 - out.array()) - y.array() / out.array()) / m;
}