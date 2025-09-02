#pragma once
#include "../Sequential.h"

class Optimizer
{
protected:
	Sequential model;
	double lr;
public:
	Optimizer(Sequential& model_, double lr_ = 0.01) : model(model_), lr(lr_) {};
	virtual ~Optimizer() = default;
	virtual void step() = 0;
	virtual void zero_grad();
};

