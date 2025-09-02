#include "../../../include/nn/optimizers/Optimizer.h"
#include "../../../include/nn/layers/Layer.h"
#include "../../../include/nn/layers/Linear.h"

void Optimizer::zero_grad()
{
	for (auto& layer : model.layers)
	{
		Linear* linear = dynamic_cast<Linear*>(layer.get());
		if (linear)
		{
			linear->dW.setZero();
			linear->db.setZero();
		}
	}
}