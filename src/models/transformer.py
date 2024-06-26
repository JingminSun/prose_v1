"""
This file contains complete transformer encoder/decoder modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_utils import (
    Embedding,
    SinusoidalPE,
    LearnablePE,
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
    OperatorDecoderLayer,
)
from logging import getLogger

logger = getLogger()

"""
Transformer Data modules

"""


class TransformerDataEncoder(nn.Module):
    """
    Encoder Transformer for data
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.dim_emb,
                nhead=config.n_head,
                dim_feedforward=config.dim_ffn,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=config.norm_first,
            ),
            num_layers=config.n_layer,
            norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
        )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        """
        x: Tensor (bs, slen, dim)
        """

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class TransformerDataDecoder(nn.Module):
    """
    Encoder-decoder Transformer for data (autoregressive)
    """

    def __init__(self, config, output_dim):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if self.config.kv_cache:
            self.transformer_decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )
        else:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, output_dim))

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
    ):
        """
        Inputs:
            tgt:    Tensor (bs, output_len, dim)
                    should be data in the range [input_len-1, max_len-1)
            memory: Tensor (bs, input_len, dim)
                    output from encoder
        Output:
            tgt_output: Tensor (bs, output_len, output_dim)
                        should corresponds to data in the range [input_len, max_len)
        """

        if self.positional_embedding is not None:
            tgt = self.positional_embedding(tgt)  # (bs, output_len, dim)

        tgt = tgt.transpose(0, 1)  # (output_len, bs, dim)
        memory = memory.transpose(0, 1)  # (input_len, bs, dim)

        if self.config.kv_cache:
            tgt_mask = None  # (causal decoder handles this automatically)
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), tgt.device)

        decoded = self.transformer_decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
        )  # (output_len, bs, dim)

        tgt_output = self.post_proj(decoded).transpose(0, 1)  # (bs, output_len, output_dim)

        return tgt_output

    def generate(
        self,
        encoded,
        initial,
        input_times,
        output_times,
        all_times,
        pre_proj,
    ):
        """
        For evaluation/testing only.
        Inputs:
            encoded:      Tensor (input_len, bs, dim)
            initial:      Tensor (bs, query_dim+data_dim)
            input_times:  Tensor (input_len, query_dim)
            output_times: Tensor (output_len, query_dim)
            all_times:    Tensor (max_len, query_dim)
            pre_proj:     Projection for data input
        Output:
            data_output:  Tensor (output_len, bs, data_dim)

        """
        cur_len = 1
        output_len = output_times.size(0)
        bs = initial.size(0)
        query_dim = output_times.size(1)
        data_dim = initial.size(1) - query_dim
        generated = torch.zeros(output_len, bs, data_dim, dtype=initial.dtype, device=initial.device)

        cache = None
        tgt = pre_proj(initial)[None]  # (1, bs, dim)

        # generation loop
        while cur_len <= output_len:  # max length of generation

            if self.config.kv_cache:
                decoded, cache = self.transformer_decoder(tgt=tgt, memory=encoded, cache=cache)  # (cur_len, bs, dim)
            else:
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), tgt.device)
                decoded = self.transformer_decoder(tgt=tgt, memory=encoded, tgt_mask=tgt_mask)  # (cur_len, bs, dim)

            new_data = self.post_proj(decoded[-1])  # (bs, data_dim)

            generated[cur_len - 1] = new_data

            new_input = torch.cat(
                [output_times[cur_len - 1][None].expand(bs, query_dim), new_data], dim=1
            )  # (bs, query_dim + data_dim)

            tgt = torch.cat([tgt, pre_proj(new_input[None])], dim=0)  # (cur_len + 1, bs, dim)

            cur_len += 1

        return generated


class DataOperatorDecoder(nn.Module):
    """
    Operator Decoder for data
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.dim = config.dim_emb

        self.time_proj = nn.Sequential(
            nn.Linear(config.query_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.GELU(),
            # nn.LayerNorm(self.dim),
        )

        # self.patch_position_embeddings = nn.Parameter(
        #     (self.dim**-0.5) * torch.randn(1, 1, config.patch_num * config.patch_num, self.dim)
        # )
        self.patch_position_embeddings = nn.Parameter(torch.randn(1, 1, config.patch_num * config.patch_num, self.dim))

        self.transformer_decoder = nn.TransformerDecoder(
            OperatorDecoderLayer(
                d_model=config.dim_emb,
                nhead=config.n_head,
                dim_feedforward=config.dim_ffn,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=config.norm_first,
            ),
            num_layers=config.n_layer,
            norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
        )

    def get_query_emb(self, times):
        """
        Input:
            times:     Tensor (bs, output_len, 1)
        Output:
            query_emb: Tensor (bs, query_len, dim)
                       query_len = output_len * patch_num * patch_num
        """

        bs, output_len, query_dim = times.size()

        times = self.time_proj(times)[:, :, None]  # (bs, output_len, 1, dim)
        # patches = self.patch_position_embeddings[None, None, :, :]  # (1, 1, p*p, dim)

        return (times + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def forward(self, src, query_emb, src_key_padding_mask=None):
        """
        src:         Tensor (bs, src_len, dim)
        query_emb:   Tensor (bs, query_len, dim)
        src_key_padding_mask: Optional[Tensor] (bs, src_len)
        """

        x = self.transformer_decoder(query_emb, src, memory_key_padding_mask=src_key_padding_mask)

        return x  # (bs, query_len, dim)


"""
Transformer Symbol Modules

"""


class TransformerSymbolEncoder(nn.Module):
    """
    Encoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.dim_emb,
                nhead=config.n_head,
                dim_feedforward=config.dim_ffn,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=config.norm_first,
            ),
            num_layers=config.n_layer,
            norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
        )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        """
        x:                    LongTensor (bs, slen)
        mask:                 Optional[Tensor] (bs, slen, slen)
        src_key_padding_mask: Optional[BoolTensor] (bs, slen)         (positions with value True will be ignored)
        """

        x = self.word_embeddings(x)  # (bs, slen, dim)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class TransformerSymbolDecoder(nn.Module):
    """
    Encoder-decoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if self.config.kv_cache:
            self.transformer_decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )
        else:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

        # output layer
        self.proj = nn.Linear(self.dim, self.n_words, bias=True)
        if config.share_inout_emb:
            self.proj.weight = self.word_embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        tgt,
        memory,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Inputs:
            tgt:                     LongTensor (bs, output_len)
            memory:                  LongTensor (bs, input_len, dim)        (output from encoder)
            tgt_key_padding_mask:    Optional[BoolTensor] (bs, output_len)  (True for positions that should be ignored)
            memory_key_padding_mask: Optional[BoolTensor] (bs, input_len)
        Output:
            decoded:                 Tensor (bs, output_len, dim)
        """

        tgt = self.word_embeddings(tgt)  # (bs, output_len, dim)

        if self.positional_embedding is not None:
            tgt = self.positional_embedding(tgt)  # (bs, output_len, dim)

        tgt = tgt.transpose(0, 1)  # (output_len, bs, dim)
        memory = memory.transpose(0, 1)  # (input_len, bs, dim)

        if self.config.kv_cache:
            tgt_mask = None  # (causal decoder handles this automatically)
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), tgt.device)

        decoded = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True,
        )  # (output_len, bs, dim)

        return decoded.transpose(0, 1)  # (bs, output_len, dim)

    def predict(self, output, pred_mask, y):
        """
        Given the last hidden state, compute word scores and the loss.
        Inputs:
            output       Tensor     (bs, output_len, dim)
            pred_mask    BoolTensor (bs, output_len), filled with 1 when we need to predict a word
            y            LongTensor (pred_mask.sum(), )
        """
        x = output[pred_mask.unsqueeze(-1).expand_as(output)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.float(), y, reduction="mean")
        return scores, loss

    '''
    ## TODO: merge two generate functions

    def generate(
        self,
        encoded,
        initial,
        output_times,
        pre_proj,
    ):
        """
        For evaluation/testing only.
        Inputs:
            encoded:      Tensor (input_len, bs, dim)
            initial:      Tensor (bs, query_dim+data_dim)
            output_times: Tensor (output_len, query_dim)
            pre_proj:     Projection for data input
        Output:
            data_output:  Tensor (output_len, bs, data_dim)

        """
        cur_len = 1
        output_len = output_times.size(0)
        bs = initial.size(0)
        query_dim = output_times.size(1)
        data_dim = initial.size(1) - query_dim
        generated = torch.zeros(output_len, bs, data_dim, dtype=initial.dtype, device=initial.device)

        cache = None
        tgt = pre_proj(initial)[None]  # (1, bs, dim)

        # generation loop
        while cur_len <= output_len:  # max length of generation

            if self.config.kv_cache:
                decoded, cache = self.transformer_decoder(tgt=tgt, memory=encoded, cache=cache)  # (cur_len, bs, dim)
            else:
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), tgt.device)
                decoded = self.transformer_decoder(tgt=tgt, memory=encoded, tgt_mask=tgt_mask)  # (cur_len, bs, dim)

            new_data = self.post_proj(decoded[-1])  # (bs, data_dim)

            generated[cur_len - 1] = new_data

            new_input = torch.cat(
                [output_times[cur_len - 1][None].expand(bs, query_dim), new_data], dim=1
            )  # (bs, query_dim + data_dim)

            tgt = torch.cat([tgt, pre_proj(new_input[None])], dim=0)  # (cur_len + 1, bs, dim)

            cur_len += 1

        return generated

    def generate(self, src_enc, src_len, max_len=200, top_p=1.0, sample_temperature=None):
        """
        Decode a sentence given initial start.
        Inputs:
            src_end     (bs, slen, dim) fused features
            src_len     tuple of data and text lengths
        Outputs:
            x           LongTensor(bs, slen)
                            <BOS> W1 W2 W3 <EOS> <PAD>
                            <BOS> W1 W2 W3   W4  <EOS>
            lengths     LongTensor(bs)
                            [5, 6]
        """

        # input batch
        data_len, text_len = src_len[0], src_len[1]  # (bs, )
        bs = data_len.size(0)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = text_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[0].fill_(self.bos_index)  # we use <EOS> for <BOS> everywhere

        # positions
        positions = text_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = text_len.clone().fill_(1)  # (bs, )
        unfinished_sents = text_len.clone().fill_(1)

        # cache compute states
        self.cache = {"slen": 0}
        while cur_len < max_len:
            # compute word scores
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],  # (cur_len, bs)
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )
            # assert tensor.size() == (cur_len, bs, self.dim)

            tensor = tensor.data[-1, :, :].to(self.dtype)  # (bs, dim)  ##BE CAREFUL
            scores = self.proj(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(
                    F.softmax(scores.float() / sample_temperature, dim=1), num_samples=1
                ).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
        # sanity check
        # assert (generated == self.eos_index).sum() == 2 * bs
        generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
        return generated[:cur_len], gen_len
    '''


"""
Transformer Fusion Module

"""


class TransformerFusion(nn.Module):
    """
    Fusion Transformer
    """

    def __init__(self, config, num_types=2):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.dim_emb,
                nhead=config.n_head,
                dim_feedforward=config.dim_ffn,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=config.norm_first,
            ),
            num_layers=config.n_layer,
            norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
        )

        if config.type_embeddings:
            self.type_embeddings = Embedding(num_types, self.dim)
        else:
            self.type_embeddings = None

    def forward(self, x0, x1, key_padding_mask0=None, key_padding_mask1=None):
        """
        x0: Tensor (bs, slen0, dim)
        x1: Tensor (bs, slen1, dim)
        key_padding_mask0: Optional[BoolTensor] (bs, slen0)           (True for positions that should be ignored)
        key_padding_mask1: Optional[BoolTensor] (bs, slen1)
        """

        bs = x0.size(0)

        if self.type_embeddings is not None:
            type0 = torch.zeros(1, 1, dtype=torch.long, device=x0.device)
            type1 = torch.ones(1, 1, dtype=torch.long, device=x1.device)
            x0 = x0 + self.type_embeddings(type0).expand_as(x0)
            x1 = x1 + self.type_embeddings(type1).expand_as(x1)

        x = torch.cat([x0, x1], dim=1)  # (bs, slen0+slen1, dim)

        if key_padding_mask0 is None and key_padding_mask1 is None:
            fused_mask = None
        else:
            if key_padding_mask0 is None:
                key_padding_mask0 = torch.zeros(bs, x0.size(1), dtype=torch.bool, device=x0.device)
            if key_padding_mask1 is None:
                key_padding_mask1 = torch.zeros(bs, x1.size(1), dtype=torch.bool, device=x1.device)
            fused_mask = torch.cat([key_padding_mask0, key_padding_mask1], dim=1)  # (bs, slen0+slen1)

        x = self.transformer_encoder(x, src_key_padding_mask=fused_mask)

        return x, fused_mask  # (bs, slen0+slen1, dim)
