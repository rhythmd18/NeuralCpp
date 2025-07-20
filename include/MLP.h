#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include "Layers.h"

class MLP
{
	Eigen::MatrixXd m_X;
public:
	MLP(Eigen::MatrixXd X) : m_X(X) 
	{

	};
};

