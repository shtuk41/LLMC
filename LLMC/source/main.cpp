#include <torch/torch.h>

#include <utils.h>

int main()
{
	try
	{
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size()); 
		std::set<char> inputSet;
		
		for (const auto it : inputBuffer)
			inputSet.insert(it);
			
		encode encodeO(inputSet);
		decode decodeO(inputSet);
	
		torch::Tensor data = torch::tensor(encodeO(inputBuffer), torch::kLong);
		
		printTensorShape(data);
		std::cout << "Data type: " << data.dtype() << std::endl;
		
		size_t trainSize = data.size(0) * 0.9;
		size_t valSize = data.size(0) - trainSize;
		
		std::cout << std::format("Train data size: {}\n", trainSize);
		std::cout << std::format("Validation data size: {}\n", valSize);
		torch::Tensor train_data = data.slice(0,0,trainSize);
		torch::Tensor val_data = data.slice(0,trainSize,trainSize + valSize);

		int block_size = 8;
		auto x = train_data.slice(0,0,block_size);
		auto y = train_data.slice(0,1,block_size + 1);
		
		std::cout << "full block: " << x << std::endl;
		
		for (int t = 0; t < block_size; t++)
		{
			std::cout << "index:: " << t << std::endl;
			torch::Tensor context = x.slice(0,0,t + 1);
			torch::Tensor target = y[t];
			std::cout << "context: " << context << std::endl;
			std::cout << "target: " << target << std::endl;
		}
		
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
		return 0;
	}
	
	torch::Tensor xenc = torch::zeros({ static_cast<long>(5), 27 }, torch::kFloat32);
	torch::Tensor yst = torch::zeros({ static_cast<long>(5), 1 }, torch::kInt64);
	
	std::cout << "The end\n";
	return 0;
}

