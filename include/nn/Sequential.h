#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "layers/Layer.h"
#include "criteria/Criterion.h"

class Sequential
{
public:
	std::vector<std::shared_ptr<Layer>> layers;

	template<typename... Layers>
	Sequential(Layers... ls)
	{
		int dummy[] = { (layers.push_back(std::make_shared<Layers>(ls)), 0)... };
		(void)dummy;
	}

	Eigen::MatrixXd operator()(const Eigen::MatrixXd& X)
	{
		Eigen::MatrixXd out = X;
		for (auto& layer : layers)
		{
			out = (*layer)(out);
		}
		return out;
	}

	void backward(Criterion& objective)
	{
		Eigen::MatrixXd dout = objective._backward();
		for (auto it = layers.rbegin(); it != layers.rend(); ++it)
		{
			dout = (*it)->_backward(dout);
		}
	}
};