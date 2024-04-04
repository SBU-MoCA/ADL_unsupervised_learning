import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

import transformers


class Embedding(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super().__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
		self.pooling = nn.AvgPool2d(kernel_size=(2, 2))
		self.word_embeddings = self.llama.get_input_embeddings().weight
		self.vocab_size = self.word_embeddings.shape[0]
		self.num_tokens = 1000
