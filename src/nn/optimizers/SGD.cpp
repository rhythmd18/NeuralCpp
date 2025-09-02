#include "../../../include/nn/optimizers/SGD.h"
#include "../../../include/nn/layers/Layer.h"
#include "../../../include/nn/layers/Linear.h"

void SGD::step()
{
	for (auto& layer : model.layers)
	{
		Linear* linear = dynamic_cast<Linear*>(layer.get());
		if (linear)
		{
			linear->V_dW = momentum * linear->V_dW.array() + (1 - momentum) * linear->dW.array();
			linear->V_db = momentum * linear->V_db.array() + (1 - momentum) * linear->db.array();

			linear->W = linear->W.array() - lr * linear->V_dW.array();
			linear->b = linear->b.array() - lr * linear->V_db.array();
		}
	}
}