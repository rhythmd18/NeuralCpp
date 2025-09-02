#include "../../../include/nn/criteria/BinaryCrossEntropyLoss.h"
#include <Eigen/Dense>

double BinaryCrossEntropyLoss::operator()(const Eigen::MatrixXd& y_, const Eigen::MatrixXd& out_)
{
	y = y_;
	out = out_;
	Eigen::Index m = y.rows();
	const double eps = 1e-7;
	Eigen::ArrayXd out_clamped = out.array().min(1.0 - eps).max(eps);
	return -1.0 * (y.array() * out_clamped.array().log() + (1 - y.array()) * (1 - out_clamped.array()).log()).sum() / m;
}

Eigen::MatrixXd BinaryCrossEntropyLoss::_backward()
{
	Eigen::Index m = y.rows();
	const double eps = 1e-7;
	Eigen::ArrayXd out_clamped = out.array().min(1.0 - eps).max(eps);
	return ((1 - y.array()) / (1 - out_clamped.array()) - y.array() / out.array()) / m;
}