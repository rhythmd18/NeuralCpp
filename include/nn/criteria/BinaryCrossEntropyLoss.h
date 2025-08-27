#pragma once
#include "Criterion.h"
#include <Eigen/Dense>

class BinaryCrossEntropyLoss : public Criterion
{
public:
	Eigen::MatrixXd y, out;

	double operator()(const Eigen::MatrixXd& y_, const Eigen::MatrixXd& out_)
	{
		y = y_;
		out = out_;
		Eigen::Index m = y.rows();
		return -(y.array() * out.array().log() + (1 - y.array()) * (1 - out.array()).log()).sum() / m;
	}

	Eigen::MatrixXd _backward() override
	{
		Eigen::Index m = y.rows();
		return ((1 - y.array()) / (1 - out.array()) - y.array() / out.array()) / m;
	}
};