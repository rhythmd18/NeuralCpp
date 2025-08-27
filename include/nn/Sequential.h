#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "layers/Layer.h"
#include "criteria/Criterion.h"

class Sequential
{
	std::vector<std::shared_ptr<Layer>> layers;
public:
	template<typename... Layers>
	Sequential(Layers... ls)
	{
		int dummy[] = { (layers.push_back(std::make_shared<Layers>(ls)), 0)... };
		(void)dummy;
	}

	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);
	void backward(Criterion& objective);
};