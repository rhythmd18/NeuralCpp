#pragma once
#include "Criterion.h"
#include <Eigen/Dense>

class BinaryCrossEntropyLoss : public Criterion
{
	Eigen::MatrixXd y, out;
public:
	double operator()(const Eigen::MatrixXd& y_, const Eigen::MatrixXd& out_) override;
	Eigen::MatrixXd _backward() override;
};