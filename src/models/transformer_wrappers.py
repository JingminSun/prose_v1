"""
Final model wrappers. 
"""

import torch
import torch.nn as nn

from .attention_utils import get_padding_mask
from .transformer import (
    TransformerDataEncoder,
    DataOperatorDecoder,
    TransformerSymbolEncoder,
    TransformerFusion,
    TransformerSymbolDecoder
)
from .embedder import LinearEmbedder_1DPDE
from logging import getLogger

logger = getLogger()



class PROSE_1DPDE(nn.Module):
    """
    Wrapper for the full PROSE model (2to1).
    For 1D PDE
    """

    def __init__(self, config, symbol_env, data_config):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = data_config.x_num
        self.max_output_dim = data_config.max_output_dimension

        self.embedder = LinearEmbedder_1DPDE(config.embedder, self.x_num, self.max_output_dim,full_tx = config.data_decoder.full_tx)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)
        self.data_decoder = DataOperatorDecoder(config.data_decoder)
        if not self.config.no_text_encoder:
            self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
            self.fusion = TransformerFusion(config.fusion)
        if not self.config.no_text_decoder:
            self.symbol_decoder = TransformerSymbolDecoder(config.symbol_decoder,symbol_env.equation_id2word)


    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if p.requires_grad]):,}\n"
        if not self.config.no_text_encoder:
            s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
            s += f"\tFusion:          {sum([p.numel() for p in self.fusion.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if p.requires_grad]):,}\n"

        if not self.config.no_text_decoder:
            s += f"\tSymbol Decoder:    {sum([p.numel() for p in self.symbol_decoder.parameters() if p.requires_grad]):,}"

        return s

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

    def fwd_data(
        self,
        data_input,
        input_times,
        output_times,
        symbol_input,
        query_space_grid = None,
        symbol_input_padding_mask=None,
    ):
        """
        Inputs:
            data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:            Tensor     (bs, input_len, 1)
            output_times:           Tensor     (bs, output_len, 1)

            symbol_input:           LongTensor           (bs, symbol_len)
            symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        output = {}
        bs, input_len, x_num,  data_dim = data_input.size()
        # symbol_len = symbol_input.size(1)
        # symbol_padding_mask = get_padding_mask(symbol_lengths)  # (bs, max_len)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num 
        """

        data_input = self.embedder.encode(data_input, input_times)  # (bs, data_len, dim)
        data_len = data_input.size(1)
        output["data_embeded"] = data_input
        """
        Step 2: Encode + Fusion
            data_input:   Tensor     (bs, data_len, dim)
            symbol_input: LongTensor (bs, symbol_len)
        """

        data_encoded = self.data_encoder(data_input)  # (bs, data_len, dim)
        if not self.config.no_text_encoder:
            symbol_encoded = self.symbol_encoder(
                symbol_input, src_key_padding_mask=symbol_input_padding_mask
            )  # (bs, symbol_len, dim)

            fused, fused_mask = self.fusion(
                x0=data_encoded,
                x1=symbol_encoded,
                key_padding_mask0=None,
                key_padding_mask1=symbol_input_padding_mask,
            )  # (bs, data_len+symbol_len, dim)

            output["symbol_encoded"] = symbol_encoded
        else:
            fused = data_encoded
            fused_mask = None
        output["data_encoded"] = data_encoded
        output["fused"] = fused
        output["fused_mask"] = fused_mask
        """
        Step 3: Decode data
        """

        query_emb = self.data_decoder.get_query_emb(output_times, query_space_grid =  query_space_grid)  # (bs, query_len_t* query_len_x, dim)

        data_output = self.data_decoder(
            src=fused, query_emb=query_emb, src_key_padding_mask=fused_mask
        )  # (bs, query_len_t * query_len_x, dim)

        if query_space_grid is not None:
            data_output = data_output.reshape(bs,output_times.shape[1], query_space_grid.shape[1],-1 )
        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, data_dim)

        output["data_output"] = data_output



        return output


    def fwd(
        self,
        data_input,
        input_times,
        output_times,
        symbol_input,
        symbol = None,
        query_space_grid = None,
        symbol_input_padding_mask=None,
        symbol_padding_mask=None,
        symbol_pred_mask = None,
        symbol_y = None
    ):
        """
        Inputs:
            data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:            Tensor     (bs, input_len, 1)
            output_times:           Tensor     (bs, output_len, 1)

            symbol_input:           LongTensor           (bs, symbol_len)
            symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        # symbol_len = symbol_input.size(1)
        # symbol_padding_mask = get_padding_mask(symbol_lengths)  # (bs, max_len)
        output = self.fwd_data(
            data_input = data_input,
            input_times = input_times,
            output_times = output_times,
            symbol_input =  symbol_input,
            query_space_grid = query_space_grid,
            symbol_input_padding_mask= symbol_input_padding_mask,
        )
        if not self.config.no_text_decoder:

            text_decoded = self.symbol_decoder(
                "fwd",
                tgt = symbol,
                memory = output["fused"],
                tgt_key_padding_mask=symbol_padding_mask,
                memory_key_padding_mask=output["fused_mask"],
            )  # (slen, bs, dim)

            _, text_loss = self.symbol_decoder(
                "predict",
                output=text_decoded,
                pred_mask=symbol_pred_mask,
                y= symbol_y
            )
            output["text_decoded"] = text_decoded
            output["text_loss"] = text_loss

        return output

    def generate(self, **kwargs):
        if self.config.no_text_decoder:
            return self.fwd(**kwargs)
        else:

            output = self.fwd_data(**kwargs)
            symbol_generate, generate_len = self.symbol_decoder(
                "generate",
                memory= output["fused"],
                memory_key_padding_mask=output["fused_mask"],
                max_len = self.config.symbol_decoder.max_generated_output_len
            )
            output["symbol_generated"] = symbol_generate
            output["generate_len"] = generate_len
            return output
