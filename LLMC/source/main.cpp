
#include <string>
#include <string_view>
#include <tuple>

#include <torch/torch.h>

#include <utils.h>

int main()
{
	torch::xpu::manual_seed(1337);
	
	int batch_size = 4;
	int block_size = 8;
	
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

		auto get_batch = [&](std::string_view split) -> std::pair<torch::Tensor, torch::Tensor>
		{
			torch::Tensor current_data;
			
			if (split == "train")
			{
				current_data = train_data;
			}
			else
			{
				current_data = val_data;
			}
			
			// Generate random starting indices for the batch
			auto ix = torch::randint(0, current_data.size(0) - block_size, {batch_size}, torch::kLong);
			auto ix_accessor = ix.accessor<int64_t, 1>();
			
			std::vector<torch::Tensor> x_list;
			std::vector<torch::Tensor> y_list;
			x_list.reserve(batch_size);
			y_list.reserve(batch_size);
			
			for (int64_t i = 0; i < batch_size; ++i)
			{
				int64_t idx = ix_accessor[i];
				x_list.push_back(current_data.slice(0, idx, idx + block_size));
				y_list.push_back(current_data.slice(0, idx + 1, idx + block_size + 1));
			}
			
			auto x = torch::stack(x_list);
			auto y = torch::stack(y_list);
			
			return {x, y};
		};
    
		// Use square brackets for C++ structured binding
		auto [xb, yb] = get_batch("train");
		
		std::cout << "xb:\n" << xb << std::endl;
		std::cout << "yb:\n" << yb << std::endl;
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
		return 0;
	}
		
	std::cout << "The end\n";
	return 0;
}

