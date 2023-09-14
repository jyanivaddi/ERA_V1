import torch
from torch.nn import functional as F
import torch.nn as nn
import math
from torch.cuda.amp import autocast as autocast

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=1E-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha, learnable mean
        self.bias = nn.Parameter(torch.zeros(1)) # bias, learnable bias

    def forward(self,x):
        # x: (batch_size, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha*(x-mean)/(std+self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 =  nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
    
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # multiply by sqrt(d_model) to scale the embeddings per the paper
        return self.embedding(x)*math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # create a vector of shape (d_model)
        # exp((-2i/d_model)*log(10000)  
        # => exp(log(10000**(-2*i/d_model))) 
        # => 10000**(-2*i/d_model) 
        # => 1/(10000**(2*i/d_model))
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.)/d_model)) 
        # Apply sin to even indices
        pe[:,0::2] = torch.sin(position*div_term)
        # Apply cos to odd indices
        pe[:,1::2] = torch.cos(position*div_term)
        # Add batch term
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the positional encoding as a buffer
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # number of heads
        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of the vector as seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias = False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        _MASKING_VALUE = -1e9 if attention_scores.dtype == torch.float32 else -1e4
        if mask is not None:
            with autocast():
                # Write a very low value (indicating -inf) to the positions where mask == 0
                attention_scores.masked_fill_(mask == 0, _MASKING_VALUE)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax to the attention outputs
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, block_size, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, self.tril[:T, :T]))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int=512,
                      N: int=6,
                      h: int=8,
                      dropout: float=0.1,
                      d_ff: int=2048,
                      parameter_sharing: bool=True) -> Transformer:
    # control d_ff to control RAM 
    # create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Halve the blocks if using parameter sharing
    num_blocks = N
    if parameter_sharing:
        num_blocks = N//2

    # create encoder blocks
    encoder_blocks = []
    for _ in range(num_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(num_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    if parameter_sharing:
        # parameter sharing
        e1, e2, e3 = encoder_blocks
        d1, d2, d3 = decoder_blocks
        encoder_blocks1 = [e1, e2, e3, e3, e2, e1]
        decoder_blocks1 = [d1, d2, d3, d3, d2, d1]
        # create encoder and decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks1))
        decoder = Decoder(nn.ModuleList(decoder_blocks1))
    else:
        # create encoder and decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))


    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total Parameters: {num_params}")
    return transformer


class BERT(nn.Module):
    def __init__(self,
                 N: int,  # number of encoder blocks
                 h: int,  # number of heads for multi head attention
                 d_model: int,  # embeddings size
                 d_ff: int,  # hidden layer dimension for the feed forward network
                 vocab_size: int,  # size of vocab
                 seq_len: int,  # max length of sequence
                 dropout=.1):
        super().__init__()

        # model input
        # self.embeddings = InputEmbeddings(d_model, vocab_size)
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, seq_len, dropout)

        # backbone
        encoder_blocks = []
        for i in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        self.encoder = Encoder(nn.ModuleList(encoder_blocks))

        # language model
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

class GPT(nn.Module):
    def __init__(self,
                 vocab_size, # size of the vocabulary
                 d_model: int,  # embeddings size
                 block_size: int,  # what is the maximum context length for predictions?
                 num_heads: int,  # number of attention heads
                 num_layers: int,  # number of decoder blocks
                 dropout: float,  # drop out probability
                 device: str,  # device to run
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = self.d_model*4
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.d_model)
        # each position from 0 to block_size-1 will get its embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.d_model)
        decoder_blocks = []
        for _ in range(self.num_layers):
            # create the decoder blocks
            decoder_self_attention_block = MultiHeadAttentionBlock(self.d_model, self.num_heads, self.dropout)
            feed_forward_block = FeedForwardBlock(self.d_model, self.d_ff, self.dropout)
            decoder_block = DecoderBlock(self.block_size, decoder_self_attention_block,
                                         feed_forward_block, self.dropout)
            decoder_blocks.append(decoder_block)

        self.blocks = nn.Sequential(*decoder_blocks)
        # we add the layer norm before the Linear layer
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B,T) tensor of integers
        # the token_emb is (B, T, C), C = NUM_EMBED
        token_emb = self.token_embedding_table(idx)
        # (T, C)
        posit_emb = self.position_embedding_table(torch.arange(T, device=self.device))

        x = token_emb + posit_emb
        # apply one head of self-attention
        x = self.blocks(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        # compute the loss
        if targets != None:
            # cross_entropy accepts inputs in a (batch_size, num_classes)
            # so we need to reformat our logits dimensions to
            # (batch_size * time, dim_vocabulary), time = block_size
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx



















