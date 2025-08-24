#pragma once
#include <Eigen/Dense>
#include <random>
#include <cassert>
#include "Layer.h"

class Linear : public nn::layers::Layer
{
	Eigen::MatrixXd W;
	Eigen::VectorXd b;
	Eigen::MatrixXd dW;
	Eigen::VectorXd db;
	Eigen::MatrixXd m_X;
	Eigen::MatrixXd m_Z;

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
		b = Eigen::VectorXd::Zero(output_dims);
	}

	Eigen::MatrixXd forward(const Eigen::MatrixXd& X)
	{
		assert(X.cols() == W.rows() && "X and W cannot be multiplied");
		m_X = X;
		m_Z = (X * W).rowwise() + b.transpose();
		return m_Z;
	}
};

