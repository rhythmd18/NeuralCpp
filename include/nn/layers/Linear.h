#pragma once
#include <Eigen/Dense>
#include "Layer.h"

class Linear : public Layer
{
	Eigen::MatrixXd W, X, Z, dW;
	Eigen::RowVectorXd b, db;

public:
	Linear(int input_dims, int output_dims);
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X_) override;
	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dZ) override;
};

