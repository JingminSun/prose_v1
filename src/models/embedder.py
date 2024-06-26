import torch
import torch.nn as nn

from logging import getLogger

logger = getLogger()


def patchify(data: torch.Tensor, patch_num: int):
    """
    Input:
        (bs, nt, px, py, d)
    Output:
        (bs, nt, p*p, x*y*d)
    """
    bs, nt, px, py, d = data.size()
    p = patch_num
    x = px // p
    y = py // p

    data = data.view(bs, nt, p, x, p, y, d).permute(
        0, 1, 2, 4, 3, 5, 6
    )  # (bs, nt, p, x, p, y, d) -> (bs, nt, p, p, x, y, d)

    data = data.reshape((bs, nt, p * p, x * y * d))
    return data

def patchify_1D(data: torch.Tensor, patch_num: int):
    """
    Input:
        (bs, nt, px, d)
    Output:
        (bs, nt, p, x*d)
    """
    bs, nt, px,  d = data.size()
    p = patch_num
    x = px // p

    data = data.view(bs, nt, p, x, d) #  (bs, nt, p, x, d)

    data = data.reshape((bs, nt, p, x  * d))
    return data

def depatchify(data: torch.Tensor, patch_num: int, x: int, y: int, d: int):
    """
    Input:
        (bs, nt, p*p, x*y*d)
    Output:
        (bs, nt, px, py, d)
    """
    bs = data.size(0)
    nt = data.size(1)
    p = patch_num

    data = data.view(bs, nt, p, p, x, y, d).permute(
        0, 1, 2, 4, 3, 5, 6
    )  # (bs, nt, p, p, x, y, d) -> (bs, nt, p, x, p, y, d)

    data = data.reshape((bs, nt, p * x, p * y, d))
    return data
def depatchify_1D(data: torch.Tensor, patch_num: int, x: int, d: int):
    """
    Input:
        (bs, nt, p, x*d)
    Output:
        (bs, nt, px, py, d)
    """
    bs = data.size(0)
    nt = data.size(1)
    p = patch_num

    data = data.view(bs, nt, p, x, d)
    data = data.reshape((bs, nt, p * x, d))
    return data


class LinearEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert x_num % config.patch_num == 0, "x_num must be divisible by patch_num"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        # for encoder part

        # self.patch_position_embeddings = nn.Parameter(
        #     (self.dim**-0.5) * torch.randn(1, 1, config.patch_num * config.patch_num, self.dim)
        # )
        self.patch_position_embeddings = nn.Parameter(torch.randn(1, 1, config.patch_num * config.patch_num, self.dim))
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.LayerNorm(self.dim, bias=False),
        )
        self.pre_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.LayerNorm(self.dim),
        )

        # for decoder part

        self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.patch_dim))

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """
        bs = data.size(0)
        data = patchify(data, self.config.patch_num)  # (bs, input_len, p*p, x*y*d)
        data = self.pre_proj(data)  # (bs, input_len, p*p, dim)

        time_embeddings = self.time_proj(times)[:, :, None]  # (bs, input_len, 1, dim)
        # patch_position_embeddings = self.patch_position_embeddings[None, None, :, :]  # (1, 1, p*p, dim)

        data = ((data + time_embeddings) + self.patch_position_embeddings).reshape(bs, -1, self.dim)

        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        data_output = self.post_proj(data_output)  # (bs, query_len, patch_dim)
        data_output = data_output.view(
            bs, -1, self.config.patch_num * self.config.patch_num, self.patch_dim
        )  # (bs, output_len, p*p, patch_dim)

        data_output = depatchify(
            data_output, self.config.patch_num, self.patch_resolution, self.patch_resolution, self.data_dim
        )  # (bs, output_len, x_num, x_num, data_dim)

        return data_output


class LinearEmbedder_1DPDE(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert x_num % config.patch_num == 0, "x_num must be divisible by patch_num"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution


        self.time_proj = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.LayerNorm(self.dim, bias=False),
        )
        self.pre_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.LayerNorm(self.dim),
        )

        # for decoder part

        self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.patch_dim))

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """
        bs = data.size(0)
        data = patchify_1D(data, self.config.patch_num)  # (bs, input_len, p, x*d)
        data = self.pre_proj(data)  # (bs, input_len, p, dim)

        time_embeddings = self.time_proj(times)[:, :, None]  # (bs, input_len, 1, dim)
        # patch_position_embeddings = self.patch_position_embeddings[None, None, :, :]  # (1, 1, p*p, dim)

        data = ((data + time_embeddings)).reshape(bs, -1, self.dim)

        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        data_output = self.post_proj(data_output)  # (bs, query_len, patch_dim)
        data_output = data_output.view(
            bs, -1, self.config.patch_num, self.patch_dim
        )  # (bs, output_len, p, patch_dim)

        data_output = depatchify_1D(
            data_output, self.config.patch_num, self.patch_resolution, self.data_dim
        )  # (bs, output_len, x_num, data_dim)

        return data_output

class LinearEmbedder_1DPDE_encode(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert x_num % config.patch_num == 0, "x_num must be divisible by patch_num"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution


        self.time_proj = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.LayerNorm(self.dim, bias=False),
        )
        self.pre_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            # nn.LayerNorm(self.dim),
        )

        # for decoder part

        # self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.patch_dim))

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """
        bs = data.size(0)
        data = patchify_1D(data, self.config.patch_num)  # (bs, input_len, p, x*d)
        data = self.pre_proj(data)  # (bs, input_len, p, dim)

        time_embeddings = self.time_proj(times)[:, :, None]  # (bs, input_len, 1, dim)
        # patch_position_embeddings = self.patch_position_embeddings[None, None, :, :]  # (1, 1, p*p, dim)

        data = ((data + time_embeddings)).reshape(bs, -1, self.dim)

        return data

class LinearEmbedder_1DPDE_decode(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert x_num % config.patch_num == 0, "x_num must be divisible by patch_num"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution

        # for decoder part

        self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.patch_dim))

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        data_output = self.post_proj(data_output)  # (bs, query_len, patch_dim)
        data_output = data_output.view(
            bs, -1, self.config.patch_num, self.patch_dim
        )  # (bs, output_len, p, patch_dim)

        data_output = depatchify_1D(
            data_output, self.config.patch_num, self.patch_resolution, self.data_dim
        )  # (bs, output_len, x_num, data_dim)

        return data_output