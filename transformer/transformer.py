import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, h: int, causal_mask: bool = False) -> None:
        super().__init__()
        self.h = h
        self.d_model = d_model 
        self.d_k = d_k
        self.d_v = d_v
        self.causal_mask = causal_mask
        self.W_q = nn.Linear(in_features=d_model, out_features=h*d_k, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=h*d_k, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=h*d_v, bias=False)
        self.W_o = nn.Linear(in_features=h*d_v, out_features=d_model, bias=False)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor = None) -> torch.Tensor:
        
        if encoder_out is not None:
            q, k, v = self.W_q(x), self.W_k(encoder_out), self.W_v(encoder_out)
            k_reshaped = k.view(x.shape[0], self.h, encoder_out.shape[1], self.d_k)
            v_reshaped = v.view(x.shape[0], self.h, encoder_out.shape[1], self.d_v)
        else:
            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
            k_reshaped = k.view(x.shape[0], self.h, x.shape[1], self.d_k)
            v_reshaped = v.view(x.shape[0], self.h, x.shape[1], self.d_v)
        q_reshaped = q.view(x.shape[0], self.h, x.shape[1], self.d_k)
        pre_soft = (q_reshaped @ k_reshaped.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k))
        if self.causal_mask:
            mask = torch.tril(pre_soft) == 0
            pre_soft = pre_soft.masked_fill(mask, float('-inf')) 
        pre_out = F.softmax(pre_soft, dim=-1) @ v_reshaped
        pre_out = pre_out.transpose(1, 2)
        pre_out_flat = pre_out.flatten(-2)
        out = self.W_o(pre_out_flat)

        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_features=d_model, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int,hidden_dim: int, d_k: int, d_v: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.hidden_dim = hidden_dim
        self.mh_att = MultiHeadAttention(d_model, d_k, d_v, h)        
        self.ffn = FeedForwardNetwork(d_model, hidden_dim)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.ln1(x + self.mh_att(x))
        x = self.ln2(x + self.ffn(x))
        
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, d_k: int, d_v: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.hidden_dim = hidden_dim
        self.masked_mh_att = MultiHeadAttention(d_model, d_k, d_v, h, True)
        self.mh_att = MultiHeadAttention(d_model, d_k, d_v, h)
        self.ffn = FeedForwardNetwork(d_model, hidden_dim)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        
        x = self.ln1(x + self.masked_mh_att(x))
        x = self.ln2(x + self.mh_att(x, encoder_out))
        x = self.ln3(x + self.ffn(x))                
        
        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, d_k: int, d_v: int, h: int, vocab_size: int, num_layers: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.encoders = nn.ModuleList([Encoder(d_model, hidden_dim, d_k, d_v, h) for _ in range(num_layers)]) 
        self.decoders = nn.ModuleList([Decoder(d_model, hidden_dim, d_k, d_v, h) for _ in range(num_layers)]) 
        self.embedding = nn.Embedding(vocab_size, d_model)        
        self.embedding.weight = nn.Parameter(self.embedding.weight * torch.sqrt(torch.tensor(d_model)))




    def forward(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        
        device = input.device
        in_seq_len, out_seq_len = input.shape[-1], output.shape[-1]
        pos = lambda seq_len: torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (- torch.log(torch.tensor(10000, device=device)) / self.d_model))
        pos_encoding = lambda seq_len: torch.zeros(1, seq_len, self.d_model, device=device)
        in_pos_encoding = pos_encoding(in_seq_len)
        out_pos_encoding = pos_encoding(out_seq_len)
        in_pos = pos(in_seq_len)
        out_pos = pos(out_seq_len)
        in_pos_encoding[0, :, 0::2] = torch.sin(in_pos * div_term) 
        in_pos_encoding[0, :, 1::2] = torch.cos(in_pos * div_term)
        out_pos_encoding[0, :, 0::2] = torch.sin(out_pos * div_term) 
        out_pos_encoding[0, :, 1::2] = torch.cos(out_pos * div_term)
        input_embed = self.embedding(input)
        output_embed = self.embedding(output)
        input_embed = input_embed + in_pos_encoding 
        output_embed = output_embed + out_pos_encoding
        encoder_out = input_embed
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out)
        decoder_out = output_embed
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out)
        out = decoder_out @ self.embedding.weight.T
        
        return out
