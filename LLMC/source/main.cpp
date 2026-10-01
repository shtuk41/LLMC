
#include <string>
#include <string_view>
#include <tuple>

#include <torch/torch.h>

#include <bigramLanguageModel.h>
#include <utils.h>

void dev1();
void dev2();
void dev3();

int main()
{
	torch::xpu::manual_seed(1337);
	
	//dev1();
	//dev2();
	dev3();
		
	std::cout << "The end\n";
	return 0;
}

void dev3()
{
	std::cout << "dev2\n\n";
	
	int batch_size = 32;
	int block_size = 8;
	
	try
	{
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size()); 
		std::set<char> inputSet;
		
		for (const auto it : inputBuffer)
			inputSet.insert(it);
		
		encode encodeO(inputSet);
		
		torch::Tensor data = torch::tensor(encodeO(inputBuffer), torch::kLong);
		
		size_t trainSize = data.size(0) * 0.9;
		size_t valSize = data.size(0) - trainSize;
		
		std::cout << std::format("Train data size: {}\n", trainSize);
		std::cout << std::format("Validation data size: {}\n", valSize);
		torch::Tensor train_data = data.slice(0, 0, trainSize);
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
    
		auto m = BigramLanguageModel(65);
		//idx = m.generate(idx, max_new_tokens);
		
		auto optimizer = torch::optim::Adam(m.parameters(), 1e-3);
		
		for (int step = 0; step < 10000; step++)
		{
			auto [xb, yb] = get_batch("train");
			auto [logits, loss] =  m.forward(xb, yb);
			
			if (step % 1000 == 0)
				std::cout << "step:  " << step << ",   " << "loss: " << loss << std::endl;
			
			optimizer.zero_grad(true);
			
			loss.backward();
			optimizer.step();
		}
		
		int max_new_tokens = 300;	
		auto idx = torch::zeros({1,1},torch::kLong);
		
		idx = m.generate(idx, max_new_tokens);
		
		auto cpu_tensor = idx.to(torch::kCPU).contiguous();
		int64_t* ptr = cpu_tensor.data_ptr<int64_t>();
		std::vector<int64_t> vec(ptr, ptr + cpu_tensor.numel());
		
		decode<int64_t> decodeO(inputSet);
		
		auto output = decodeO(vec);
		
		for (char c : output)
		{
			std::cout << c;
		}
		
		std::cout << std::endl;
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

void dev2()
{
	std::cout << "dev2\n\n";
	
	try
	{
		int max_new_tokens = 100;	
		auto idx = torch::zeros({1,1},torch::kLong);
		
		auto m = BigramLanguageModel(65);
		idx = m.generate(idx, max_new_tokens);
		
		auto cpu_tensor = idx.to(torch::kCPU).contiguous();
		int64_t* ptr = cpu_tensor.data_ptr<int64_t>();
		std::vector<int64_t> vec(ptr, ptr + cpu_tensor.numel());
		
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size()); 
		std::set<char> inputSet;
		
		for (const auto it : inputBuffer)
			inputSet.insert(it);
		
		decode<int64_t> decodeO(inputSet);
		
		auto output = decodeO(vec);
		
		for (char c : output)
		{
			std::cout << c;
		}
		
		std::cout << std::endl;
		
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

void dev1()
{
	std::cout << "dev1\n\n";
	
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
		
		auto m = BigramLanguageModel(65);
		auto [logits, loss] =  m.forward(xb, yb);
		
		std::cout << "Logits sizes: \n";
		printTensorShape(logits);
		
		//std::cout << "logits:\n" << logits << std::endl;
		std::cout << "loss:\n" << loss << std::endl;
		
		auto idx = xb;
		
		idx = m.generate(idx, 1);
		std::cout << "New ids: \n";
		std::cout << idx << std::endl;
		
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

