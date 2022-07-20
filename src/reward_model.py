from math import e, sin
from random import betavariate, gauss, randint, random
from typing import Callable

from numpy.random import binomial


def clamp(value :float, minimum :float, maximum :float) -> float:
    """clamps value between min and max"""
    return min(max(minimum, value),maximum)

# some reward functions, return (actual function, string representation)
def exp(a :int, b :int, c=1) -> "tuple[Callable, str]":
    return lambda x : clamp( c * (1 - e ** - (a * x / b)), 0, 1 ), f"{c}*(1 - e**(-{a}*x/{b}))"

def poly(rho :float, b :int, c=1) -> "tuple[Callable, str]":
    return lambda x: clamp( c * (1 - b  / ((x + b**(1/rho)) ** rho) ), 0, 1) , f"{c}*(1 - {b} / (x+{b}**(1/{rho}))**{rho})"

def beta_params(mean :float, beta :float) -> "tuple[float,float]":
    """returns the parameters alpha, beta of a beta distribution with mean = mean and beta = beta"""
    if mean <= 0:
        mean = 10**-6   # small value
    if mean >= 1:
        mean = 1 - 10**-6   # big value
    return (mean * beta) / (1 - mean), beta


class RewardModel:
    """definition of the reward (and observation) generating function"""

    # reward functions
    exponential = exp
    polynomial = poly
    sinusoidal = sin

    # reward noises
    gauss_noise = {
        "type" : "gaussian",
        "param" : 0.1
    }
    beta_noise = {
        "type" : "beta",
        "param" : 2
    }
    bernoulli_noise = {
        "type" : "bernoulli",
        "param" : None
    }
    noiseless = {
        "type" : "default",
        "param" : None
    }


    def load_model(self, reward_function: Callable, noise: dict, description: str) -> None:
        """parametrize the RewardModel with given reward function"""
        self.reward_function = reward_function

        self.noise_type = noise["type"]
        self.noise_param = noise["param"]    # sigma for gaussian distrib or beta for beta distrib
        self.observation_function = {
            "gaussian" : lambda x : self.reward_function(x) + gauss(0, self.noise_param),
            "beta" : lambda x : betavariate(*beta_params(self.reward_function(x), self.noise_param)),
            "bernoulli" : lambda x : binomial(1, clamp(self.reward_function(x), 0,1)),
            "default" : self.reward_function
        }.get(self.noise_type, "default")

        self.description = description


    def random_model(self, T, only_exp = False, only_poly = False, rewards_reach_1 = False) -> None:
        """create a random model"""
        a = randint(1, 5)
        b = randint(T, 10*T)
        rho = max(1e-1,random())
        c = 1 if rewards_reach_1 else random()

        assert not (only_poly and only_exp)
        choice = randint(0,1) if not (only_exp or only_poly) else int(only_poly)
        f, d = {
            0 : RewardModel.exponential(a,b,c),
            1 : RewardModel.polynomial(rho,b,c)
        }[choice]

        self.load_model(f, RewardModel.gauss_noise, d)
        

    def __str__(self) -> str:
        return self.description
