#pragma once

#include <torch/torch.h>

class Head : public torch::nn::Module
{
private:

	torch::nn::Linear key;
	torch::nn::Linear query;
	torch::nn::Linear value;
	torch::Tensor tril;
	int64_t head_size;

public:
	Head(int64_t head_size, int64_t block_size, int64_t n_embd) :
		key(register_module("key", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		query(register_module("query", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		value(register_module("value", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		tril(register_buffer("tril", torch::tril(torch::ones({ block_size, block_size })))),
		head_size(head_size)
	{
	}

	torch::Tensor operator()(torch::Tensor x)
	{
		return forward(x);
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor targets = torch::Tensor())
	{
		auto B = x.sizes()[0];
		auto T = x.sizes()[1];
		auto C = x.sizes()[2];

		auto k = key(x);
		auto q = query(x);

		auto wei = q.matmul(k.transpose(-2, -1)) * sqrt(head_size);
		wei = wei.masked_fill(tril == 0, -std::numeric_limits<float>::infinity());
		wei = torch::nn::functional::softmax(wei, -1);
		auto v = value(x);
		auto out = wei.matmul(v);

		return out;
	}
};