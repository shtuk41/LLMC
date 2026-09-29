#include <torch/torch.h>
#include <iostream>

int main()
{
	torch::Tensor xenc = torch::zeros({ static_cast<long>(5), 27 }, torch::kFloat32);
	torch::Tensor yst = torch::zeros({ static_cast<long>(5), 1 }, torch::kInt64);

}
