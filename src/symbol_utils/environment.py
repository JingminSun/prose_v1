from logging import getLogger
import numpy as np
from .data_generators import RandomFunctions
from collections import defaultdict

import torch

SPECIAL_WORDS = [
    "<BOS>",
    "<EOS>",
    # "<INPUT_PAD>",
    # "<OUTPUT_PAD>",
    "<PAD>",
    "<PLACEHOLDER>",
    # "(",
    # ")",
]
logger = getLogger()

SKIP_ITEM = "SKIP_ITEM"


class SymbolicEnvironment:

    def __init__(self, params):
        self.params = params
        self.rng = None
        self.seed = None
        self.float_precision = params.symbol.float_precision
        self.mantissa_len = params.symbol.mantissa_len
        self.max_size = None
        self.output_dim = params.data.max_output_dimension
        self.float_tolerance = 10 ** (-params.symbol.float_precision)
        self.additional_tolerance = [10 ** (-i) for i in range(params.symbol.float_precision + 1)]
        assert (params.symbol.float_precision + 1) % params.symbol.mantissa_len == 0, "Bad precision/mantissa len ratio"

        self.generator = RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        self.equation_words += self.float_words

        # number of words / indices
        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}

        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.symbol.n_words = len(self.equation_words)
        logger.info(f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words")

    def word_to_idx(self, words, float_input=True):
        if float_input:
            return [[torch.LongTensor([self.float_word2id[dim] for dim in point]) for point in seq] for seq in words]
        else:
            return [torch.LongTensor([self.equation_word2id[w] for w in eq]) for eq in words]

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            if self.params.symbol.use_sympy:
                try:
                    m = self.equation_encoder.sympy_decode(words)
                except:
                    m = "EMPTY"
            else:
                m = self.equation_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def idx_to_infix(self, lst, is_float=True, str_array=True):
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            idx_to_words = [self.equation_id2word[int(term)] for term in lst]
        return self.word_to_infix(idx_to_words, is_float, str_array)

    def gen_expr(self,rng):
        errors = defaultdict(int)
        while True:
            try:
                expr, error = self._gen_expr(rng)
                if error:
                    errors[error[0]] += 1
                    assert False
                return expr, errors
            except:
                if self.params.debugging:
                    print(error)
                continue

    def _gen_expr(self,rng):
        item = self.generator.generate_one_sample(rng)

        tree = item["tree"]

        if len(item["data"]) == 0:
            return item, ["data generation error"]

        if "tree_encoded" not in item:
            if self.params.symbol.use_sympy or self.params.symbol.all_type:
                try:
                    tree_sympy = item["tree_sympy"]
                    tree_encoded = self.equation_encoder.sympy_encoder(tree_sympy)
                    assert all([x in self.equation_word2id for x in
                                tree_encoded]), "tree: {}\n encoded: {}".format(
                        tree_sympy, tree_encoded
                    )
                    item["tree_encoded_sympy"] = tree_encoded
                except:
                    tree_encoded = self.equation_encoder.sympy_encoder(tree)
                    assert all([x in self.equation_word2id for x in
                                tree_encoded]), "tree: {}\n encoded: {}".format(
                        tree, tree_encoded
                    )
                    item["tree_encoded"] = tree_encoded
            if not self.params.symbol.use_sympy or self.params.symbol.all_type:
                tree_encoded = self.equation_encoder.encode(tree)
                assert all([x in self.equation_word2id for x in tree_encoded]), "tree: {}\n encoded: {}".format(
                    tree, tree_encoded
                )
                item["tree_encoded"] = tree_encoded

        return item, []

