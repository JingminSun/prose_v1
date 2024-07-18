import torch
import numpy as np
from numpy.polynomial import polynomial as P
from logging import getLogger
from scipy.integrate import solve_ivp

logger = getLogger()

from symbol_utils.node_utils import Node, NodeList


class Generator:
    def __init__(self, params, float_encoder, equation_encoder, t_span, t_eval, x_grid, dt, dx):
        self.params = params
        self.float_encoder = float_encoder
        self.equation_encoder = equation_encoder
        self.t_span = t_span
        self.t_eval = t_eval
        self.x_grid = x_grid
        self.dt = dt
        self.dx = dx

        self.ICs_per_equation = params.IC_per_param
        self.eval_ICs_per_equation = self.ICs_per_equation // 5
        self.rtol = 1e-5
        self.atol = 1e-6

        self.ph = "<PLACEHOLDER>"
        self.tree_skeletons = dict()

        self.type_to_dim = {
            "thomas": 3,
            "lorenz_3d": 3,
            "aizawa": 3,
            "chen_lee": 3,
            "dadras": 3,
            "rossler": 3,
            "halvorsen": 3,
            "fabrikant": 3,
            "sprott_B": 3,
            "sprott_linz_F": 3,
            "four_wing": 3,
            "lorenz_96_4d": 4,
            "lorenz_96_5d": 5,
            "duffing": 3,
            "double_pendulum": 4,
        }



        if self.params.symbol.noisy_text_input:
            p = self.params
            self.missing_locations = dict()
            self.addition_locations = dict()

            # generate terms to be added (polynomials of degree at most 2)
            self.addition_terms = dict()
            for dim in range(self.params.min_output_dimension, self.params.max_output_dimension + 1):
                cur_addition_terms = [Node(self.ph, p)]

                for i in range(dim):
                    cur_addition_terms.append(Node("mul", p, [Node(self.ph, p), Node(f"u_{i}", p)]))

                    for j in range(i, dim):
                        cur_addition_terms.append(
                            Node(
                                "mul",
                                p,
                                [
                                    Node(self.ph, p),
                                    Node("mul", p, [Node(f"u_{i}", p), Node(f"u_{j}", p)]),
                                ],
                            )
                        )
                self.addition_terms[dim] = cur_addition_terms

    def get_sample_range(self, mean):
        """
        Generate interval for sample parameters
        """
        gamma = self.params.data.param_range_gamma
        half_range = np.abs(mean) * gamma
        return [mean - half_range, mean + half_range]

    def get_skeleton_tree(self, type, mode=0, rng=None):
        """
        Generate skeleton tree for text input, with possibly added/deleted terms
        """
        if mode == 0:
            # no text noise
            if type not in self.tree_skeletons:
                op_list, term_list = getattr(self, type + "_tree_list")()
                tree = self.tree_from_list(op_list, term_list)
                tree_skeleton = self.equation_encoder.encode_with_placeholder(tree)
                self.tree_skeletons[type] = tree_skeleton
            return self.tree_skeletons[type]
        elif mode == -1:
            # term deletion
            assert rng is not None
            if type == "double_pendulum":
                return self.double_pendulum_missing_term(type, rng)
            else:
                return self.tree_with_missing_term(type, rng)
        elif mode == 1:
            # term addition
            assert rng is not None
            if type == "double_pendulum":
                return self.double_pendulum_additional_term(type, rng)
            else:
                return self.tree_with_additional_term(type, rng)
        else:
            assert False, "Unknown mode {}".format(mode)

    def tree_with_additional_term(self, type, rng):
        op_list, term_list = getattr(self, type + "_tree_list")()

        term_to_add = rng.choice(self.addition_terms[self.type_to_dim[type]])

        if type not in self.addition_locations:
            coords = []
            for i in range(len(term_list)):
                lst = term_list[i]
                for j in range(len(lst) + 1):
                    coords.append((i, j))
            self.addition_locations[type] = coords

        coords = self.addition_locations[type]
        coord = coords[rng.choice(len(coords))]
        i, j = coord[0], coord[1]
        if j == 0:
            if term_list[i][0].value == "neg":
                term_list[i][0] = term_list[i][0].children[0]
                op_list[i].insert(0, "sub")
                term_list[i].insert(0, term_to_add)
            else:
                op_list[i].insert(0, "add")
                term_list[i].insert(0, term_to_add)
        else:
            op_list[i].insert(j - 1, "add")
            term_list[i].insert(j, term_to_add)
        tree = self.tree_from_list(op_list, term_list)
        return self.equation_encoder.encode_with_placeholder(tree)

    def tree_with_missing_term(self, type, rng):
        op_list, term_list = getattr(self, type + "_tree_list")()

        if type not in self.missing_locations:
            coords = []
            for i in range(len(term_list)):
                lst = term_list[i]
                if len(lst) <= 1:
                    continue
                for j in range(len(lst)):
                    coords.append((i, j))
            self.missing_locations[type] = coords

        coords = self.missing_locations[type]
        coord = coords[rng.choice(len(coords))]
        i, j = coord[0], coord[1]

        if j == 0:
            op = op_list[i].pop(j)
            term_list[i].pop(j)
            if op == "sub":
                term = term_list[i][0]
                term_list[i][0] = Node("neg", self.params, [term])
        else:
            op = op_list[i].pop(j - 1)
            term_list[i].pop(j)

        tree = self.tree_from_list(op_list, term_list)
        return self.equation_encoder.encode_with_placeholder(tree)

    def refine_floats(self, lst):
        """
        Refine floats to specified precision
        """
        return np.array(self.float_encoder.decode(self.float_encoder.encode(lst)))

    def poly_tree(self, degree, var, params=None):
        """
        Generate a tree containing a polynomial with given degree and variable
        """
        assert degree >= 1
        tree = Node(var, params)
        for _ in range(degree - 1):
            tree = Node("mul", params, [Node(var, params), tree])
        return tree

    def mul_terms(self, lst):
        """
        Generate a tree containing multiplication of terms in lst
        """
        p = self.params
        tree = None
        for i in reversed(range(len(lst))):
            if tree is None:
                tree = Node(lst[i], p)
            else:
                tree = Node("mul", p, [Node(lst[i], p), tree])
        return tree

    def add_terms(self, lst):
        """
        Generate a tree containing addition of terms in lst
        """
        p = self.params
        tree = None
        for i in reversed(range(len(lst))):
            if tree is None:
                tree = lst[i]
            else:
                tree = Node("add", p, [lst[i], tree])
        return tree

    def tree_from_list(self, op_list, term_list):
        """
        Generate a tree from the operator list and term list
        """
        p = self.params
        res = []
        dim = len(op_list)
        for i in range(dim):
            ops = op_list[i]
            terms = term_list[i]
            assert len(ops) + 1 == len(terms)
            tree = None
            for j in range(len(terms)):
                if tree is None:
                    tree = terms[j]
                else:
                    tree = Node(ops[j - 1], p, [tree, terms[j]])
            res.append(tree)

        return NodeList(res)



