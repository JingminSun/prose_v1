import numpy as np
import copy
from logging import getLogger
from symbol_utils import encoders

logger = getLogger()

from symbol_utils.node_utils import math_constants


operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
    # "abs": 1,
    "inv": 1,
    # "sqrt": 1,
    # "log": 1,
    # "exp": 1,
    "sin": 1,
    # "arcsin": 1,
    "cos": 1,
    # "arccos": 1,
    # "tan": 1,
    # "arctan": 1,
    "pow": 2,
    "pow2": 1,
    "pow3": 1,
}

operators_extra = dict()

all_operators = {**operators_real, **operators_extra}


class RandomFunctions:
    def __init__(self, params, special_words):
        self.params = params
        self.max_int = params.max_int

        # self.min_output_dimension = params.min_output_dimension
        # self.min_input_dimension = params.min_input_dimension
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        self.operators = copy.deepcopy(operators_real)

        self.unaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 1]

        self.binaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 2]

        self.constants = [str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0]
        self.constants += math_constants
        self.variables = (
            ["rand"]
            + [f"u_{i}" for i in range(self.max_output_dimension)]
            # + [f"x_{i}" for i in range(self.max_input_dimension)]
            + [f"ut_{i}" for i in range(self.max_output_dimension)]
            + [f"utt_{i}" for i in range(self.max_output_dimension)]
            + [f"ux_{i}" for i in range(self.max_output_dimension)]
            + [f"uxx_{i}" for i in range(self.max_output_dimension)]
            + [f"uxxx_{i}" for i in range(self.max_output_dimension)]
            + [f"uxxxx_{i}" for i in range(self.max_output_dimension)]
            + ["x"]
        )
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )
        self.constants.remove("CONSTANT")

        self.general_encoder = encoders.GeneralEncoder(params, self.symbols, all_operators)
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words

    # def generate_float(self, rng, exponent=None):
    #     sign = rng.choice([-1, 1])
    #     mantissa = float(rng.choice(range(1, 10**self.params.float_precision)))
    #     min_power = -self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
    #     max_power = self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
    #     if not exponent:
    #         exponent = rng.randint(min_power, max_power + 1)
    #     constant = sign * (mantissa * 10**exponent)
    #     return str(constant)

    # def generate_int(self, rng):
    #     return str(rng.choice(self.constants + self.extra_constants))
