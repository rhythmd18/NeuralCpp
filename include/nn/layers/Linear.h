#pragma once
#include <Eigen/Dense>
#include "Layer.h"

class Linear : public Layer
{
public:
	Eigen::MatrixXd W, X, Z, dW, V_dW;
	Eigen::RowVectorXd b, db, V_db;
	Linear(int input_dims, int output_dims);
	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X_) override;
	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dZ) override;
};

