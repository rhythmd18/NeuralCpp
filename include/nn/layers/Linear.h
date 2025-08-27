#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cassert>
#include "Layer.h"

class Linear : public Layer
{
	Eigen::MatrixXd W, X, Z, dW;
	Eigen::RowVectorXd b, db;


public:
	Linear(int input_dims, int output_dims)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<> dist(0.0, 1.0);
		W = Eigen::MatrixXd::NullaryExpr(
			input_dims,
			output_dims,
			[&]() { return dist(gen) * std::sqrt(1.0 / input_dims); });
		b = Eigen::RowVectorXd::Zero(output_dims);
	}

	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X_) override
	{
		//std::cout << "X_.cols() = " << X_.cols() << ", W.rows() = " << W.rows() << std::endl;
		assert(X_.cols() == W.rows() && "X and W cannot be multiplied");
		X = X_;
		Z = (X * W).rowwise() + b;
		return Z;
	}

	Eigen::MatrixXd _backward(const Eigen::MatrixXd& dZ) override
	{
		dW = X.transpose() * dZ / X.rows();
		db = dZ.colwise().sum() / X.rows();
		return dZ * W.transpose();
	}
};

