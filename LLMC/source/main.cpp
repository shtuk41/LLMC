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
			
		std::cout << std::format("Number of different characters is: {}\n", inputSet.size());
		
		encode encodeO(inputSet);
		decode decodeO(inputSet);
		
		auto encodeVector = encodeO("hii there");
		
		for (auto it : encodeVector)
		{
			std::cout << it << ",";
		}
		
		std::cout << std::endl;
		
		auto decodeVector = decodeO(encodeVector);
		
		for (auto it : decodeVector)
		{
			std::cout << it;
		}
		
		std::cout << std::endl;
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

