import clip.model
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class LearnablePrompt(nn.Module):
    """
    A trainable prompt representation for the Instruct-Pix2Pix model.
    We want to learn a set of parameters that can be used to generate
    a text prompt (both negative and positive) for the diffusion model. 
    The parameters are optimized during training to a specific task.
    """
    def __init__(self, device:str , 
                 clip_model:clip.model.CLIP , 
                 template:str = "Make the image: ", 
                 ctx_len=10):
        
        super().__init__()
        self.device = device
        self.template = template
        self.clip_model = clip_model
        self.tokenizer = clip.tokenize
        
        # Freezing the text encoder - we're only learning the parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            

        self.ctx_len = ctx_len
        self.max_length = 77
        self.embedding_dim = self.clip_model.token_embedding.weight.shape[1]
        self.dtype = self.clip_model.dtype

        # Initialize context parameters
        self.context = nn.Parameter(torch.randn(1, ctx_len, self.embedding_dim, device=self.device, dtype=self.dtype ) * 0.02)

        # Pre-compute template tokens
        self.template_tokens = self.tokenizer(self.template).to(self.device)

        # Get the position of the EOS token in the template
        mask = self.template_tokens[0] == 49407
        eos_idx = int(torch.nonzero(mask).squeeze())

        self.prefix_embed = self.clip_model.token_embedding(self.template_tokens)[:, :eos_idx].detach().type(self.dtype) #[SOS] + template
        self.eos_embed = self.clip_model.token_embedding(self.template_tokens)[:, eos_idx:].detach().type(self.dtype)    #[EOS]

        # Calculate total sequence length
        self.total_length = eos_idx + ctx_len + 1  # SOS + template + ctx + EOS

        #This is just a mask for getting the position of the eos token
        self.tokenized_prompt = torch.zeros(1, self.max_length, device = self.device)
        self.tokenized_prompt[: , self.total_length - 1] = 1

        if self.total_length > self.max_length:
            print(
                f"Warning: Total sequence length ({self.total_length}) exceeds maximum length ({self.max_length}). "
                "Sequence will be truncated."
            )
        
    def get_params_snapshot(self):
        """Get the current parameter values"""
        return self.context.detach().cpu().numpy()
    
    def _encode_prompt(self, full_embeddings):
        full_embeddings = full_embeddings + self.clip_model.positional_embedding.type(self.dtype)
        full_embeddings = full_embeddings.permute(1, 0, 2) # [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
        full_embeddings = self.clip_model.transformer(full_embeddings)
        full_embeddings = full_embeddings.permute(1, 0, 2) # [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
        full_embeddings = self.clip_model.ln_final(full_embeddings) #.type(self.dtype)

        return full_embeddings
    
    def _build_full_embeddings(self):
        # Combine all embeddings: [SOS] + template + ctx + [EOS]
        return torch.cat([
            self.prefix_embed,
            self.context,
            self.eos_embed,
        ], dim=1)[:, :77]
        
    def forward(self):
        """
        Returns the pre-computed text embeddings with structure: [SOS] + template + ctx + [EOS]
        
        Returns:
            text_embeddings (torch.Tensor): Complete text embeddings
        """
        # Combine all embeddings: [SOS] + template + ctx + [EOS]
        full_embeddings = self._build_full_embeddings()
        # Encode the prompt
        output = self._encode_prompt(full_embeddings)
        
        return output