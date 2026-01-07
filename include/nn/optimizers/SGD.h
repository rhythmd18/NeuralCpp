#pragma once
#include "Optimizer.h"

class SGD : public Optimizer
{
	double momentum = 0.0;
public:
	SGD(Sequential& model, double lr = 0.01, double momentum_ = 0.0) :
		Optimizer(model, lr), momentum(momentum_) {};

	~SGD() override = default;
	void step() override;
};

