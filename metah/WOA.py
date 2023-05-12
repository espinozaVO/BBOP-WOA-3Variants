from metah.abstractMetaheuristic import AbstractMetaheuristic as absMeta
from metah.Problems import Problem
import numpy as np


class WOA(absMeta):
    """ Whale Optimization Algorithm.

    This receives a mono-objetive optimization problem and run WOA algorithm.

    Attributes:
    -----------
        a : float
            Linearly decreased value from 2 to 0.
        b : float

            A constant for defining the shape of the logarithmic spiral.
        decreased_factor : float
            How much reduces a value in each iteration.
        ci_A : float
            Coefficient A to weight the distance between X*(t) and X_i(t).
        ci_C : float
            Coefficient C to weight X*(t).

    Methods:
    --------
        update_coefficients
            Compute and update coefficients A and C
    """
    INFO = {
        'author': 'Mirjalili',
        'year': 2016,
        'name': 'Whale Optimization Algorithm',
        'short': 'WOA'
    }

    def __init__(self, population_size: int, max_iterations: int = None, budget: int = None,
                 problem: Problem = None, seed=None, a_value: float = 2, b_value=1) -> None:
        """
        Parameters:
        -----------
            population_size : int
                Number of solution evaluated together every iteration.
            max_iterations : int
                Maximum number of iteration.
            problem : Object
                The problem to optimize.
            seed : Int
                Number to initialize RNG
            a_value : float
                Linearly decreased value
            b_value : float
                A shape constant
        """
        super().__init__(population_size, max_iterations, budget, problem, seed)
        self.a = a_value
        self.b = b_value
        self.ci_C = None
        self.ci_A = None
        if budget:
            iterations = (self.budget // self.population_size) + 1
            self.step_a = self.a / iterations
        else:
            self.step_a = self.a / self.max_iterations

    def metaheuristic(self):
        """Main Algorithm WOA."""
        for i in range(self.population_size):
            # Iterate over each solution
            self.update_A()

            if self.r() < 0.5:
                self.update_C()
                if np.abs(self.ci_A[i]) < 1:
                    distance = np.abs(self.ci_C[i] * self.best_global_solution.vector - self.population[i])
                    self.population[i] = self.best_global_solution.vector - self.ci_A[i] * distance

                else:
                    j = self.get_partner_index(i)
                    distance = np.abs(self.ci_C[i] * self.population[j] - self.population[i])
                    self.population[i] = self.population[j] - self.ci_A[i] * distance

            else:
                distance = np.abs(self.best_global_solution.vector - self.population[i])
                l_value = np.random.uniform(-1, 1)
                self.population[i] = distance * np.exp(self.b * l_value) * np.cos(2 * np.pi * l_value) + \
                                     self.best_global_solution.vector

            # Bound Correction
            self.population = self.boundary_correction(self.population)
        self.evaluate_population()
        self.a -= self.step_a
        # print(self.best_global_solution.value)

    def update_A(self) -> None:
        self.ci_A = np.random.uniform(-self.a, self.a, self.population_size)

    def update_C(self) -> None:
        self.ci_C = np.random.uniform(0, 2, self.population_size)
