import numpy as np
import copy
from logging import getLogger
from .encoders import GeneralEncoder

logger = getLogger()

from .node_utils import math_constants
from generator.pde_generator import PDEGenerator

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
        self.generating_mode = params.generating_mode
        self.max_input_dimension = params.data.max_input_dimension
        self.max_output_dimension = params.data.max_output_dimension
        self.operators = copy.deepcopy(operators_real)

        self.ICs_per_equation = self.params.IC_per_param
        self.t_span = [params.data.t_range[0], params.data.t_range[1]]

        self.dt =( params.data.t_range[1] -  params.data.t_range[0])/ params.data.t_num
        self.t_eval = np.linspace(params.data.t_range[0], params.data.t_range[1], params.data.t_num + 1)[:-1]
        self.space_dim = params.data.space_dim
        if self.space_dim > 0:
            self.dx = ( params.data.x_range[1] -  params.data.x_range[0]) / params.data.x_num
            x_grid_1d = np.linspace( params.data.x_range[0], params.data.x_range[1], params.data.x_num + 1)
            x_grid_1d = x_grid_1d[:-1] + 0.5 * self.dx
            self.x_grid_size = params.data.x_num ** self.space_dim
            tmp_grids = [x_grid_1d for _ in range(self.space_dim)]

            self.x_grid = np.stack(np.meshgrid(*tmp_grids, indexing="ij"))  # (space_dim, x_num, ..., x_num)
            self.x_grid = np.moveaxis(self.x_grid, 0, -1)  # (x_num, ..., x_num, space_dim)
        else:
            self.x_grid = None
            self.dx = 0
        self.max_int = params.symbol.max_int

        self.unaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 1]

        self.binaries = [o for o in self.operators.keys() if np.abs(self.operators[o]) == 2]

        self.constants = [str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0]
        self.constants += math_constants
        self.variables = (
            ["rand"]
            + [f"u_{i}" for i in range(self.max_output_dimension)]
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

        self.general_encoder = GeneralEncoder(params.symbol, self.symbols, all_operators)
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words

        if self.generating_mode:
            self.generator = PDEGenerator(
                self.params,
                self.float_encoder,
                self.equation_encoder,
                self.t_span,
                self.t_eval,
                self.x_grid,
                self.dt,
                self.dx,
            )



    def generate_float(self, rng, exponent=None):
        sign = rng.choice([-1, 1])
        mantissa = float(rng.choice(range(1, 10**self.params.float_precision)))
        min_power = -self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        max_power = self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        if not exponent:
            exponent = rng.randint(min_power, max_power + 1)
        constant = sign * (mantissa * 10**exponent)
        return str(constant)

    def generate_one_sample(self, rng,  type=None):
        assert self.generating_mode
        return self.generator.generate_sample(rng,  type=type)

