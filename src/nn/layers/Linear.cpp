#include <Eigen/Dense>
#include <random>
#include <cassert>
#include "../../../include/nn/layers/Linear.h"

Linear::Linear(int input_dims, int output_dims)
{
	std::random_device rd;
	std::mt19937 gen(rd());

	// Xavier initialization
	double fan_in = input_dims;
	double fan_out = output_dims;
	double std = std::sqrt(2.0 / (fan_in + fan_out));
	std::normal_distribution<> dist(0.0, std);
	
	W = Eigen::MatrixXd::NullaryExpr(
		input_dims,
		output_dims,
		[&]() { return dist(gen); });
	
	b = Eigen::RowVectorXd::Zero(output_dims);
	V_dW = Eigen::MatrixXd::Zero(input_dims, output_dims);
	V_db = Eigen::RowVectorXd::Zero(output_dims);
}

Eigen::MatrixXd Linear::operator()(const Eigen::MatrixXd& X_)
{
	assert(X_.cols() == W.rows() && "X and W cannot be multiplied");
	X = X_;
	Z = (X * W).rowwise() + b;
	return Z;
}

Eigen::MatrixXd Linear::_backward(const Eigen::MatrixXd& dZ)
{
	dW = X.transpose() * dZ / X.rows();
	db = dZ.colwise().sum() / X.rows();
	return dZ * W.transpose();
}
