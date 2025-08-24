#pragma once
#include "Layer.h"
#include <Eigen/Dense>

class Tanh : nn::layers::Layer
{
	Eigen::MatrixXd m_A;

public:
	Eigen::MatrixXd forward(const Eigen::MatrixXd& X)
	{
		m_A = X.array().tanh().matrix();
		return m_A;
	}
};