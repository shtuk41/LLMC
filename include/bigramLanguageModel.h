
#pragma once

#include <torch/torch.h>

using namespace torch::indexing;

class BigramLanguageModel : public torch::nn::Module
{
private:
	torch::nn::Embedding token_embedding_table;
public:
	BigramLanguageModel(int64_t vocab_size) : token_embedding_table(register_module("token_embedding_table", torch::nn::Embedding(vocab_size, vocab_size)))
	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = torch::Tensor())
	{
		auto logits = token_embedding_table(idx);

        torch::Tensor loss = torch::Tensor();
        if (targets.defined()) 
        {
            int64_t B = logits.size(0);
            int64_t T = logits.size(1);
            int64_t C = logits.size(2);

            auto logits_flat = logits.view({B * T, C});
            auto targets_flat = targets.view({B * T});

            loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
        }

        return {logits, loss};
	}
	
	torch::Tensor generate(torch::Tensor idx, int max_new_tokens)
	{
		for (int ii = 0; ii < max_new_tokens; ii++)
		{
			auto [logits, loss] =  forward(idx);
			
			auto last_logits = logits.index({Slice(), -1, Slice()}); 
			
			auto probs = torch::nn::functional::softmax(last_logits, -1);
			
			auto idx_next = torch::multinomial(probs, 1);
			
			idx = torch::cat({idx, idx_next}, 1);
		}
		
		return idx;
	}
};
