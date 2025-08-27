#pragma once

class Criterion
{
public:
	virtual ~Criterion() = default;
	virtual double operator()(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) = 0;
	virtual Eigen::MatrixXd _backward() = 0;
};