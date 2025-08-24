#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class Sigmoid : public nn::layers::Layer
{
	Eigen::MatrixXd m_A;

public:
	Eigen::MatrixXd forward(const Eigen::MatrixXd& X)
	{
		m_A = 1.0 / (1.0 + X.array().exp());
		return m_A;
	}
};