from metah.abstractMetaheuristic import AbstractMetaheuristic as absMeta
from metah.Problems import Problem
import numpy as np
import time


class E_WOA(absMeta):
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
        'author': 'Nadimi',
        'year': 2022,
        'name': 'Enhanced Whale Optimization Algorithm',
        'short': 'E-WOA'
    }

    def __init__(self, population_size: int, max_iterations: int = None, budget: int = None,
                 problem: Problem = None, seed=None, b_value=1, kappa=1.5, p_rate=20) -> None:
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
        self.b = b_value
        self.kappa = int(kappa * self.population_size)
        self.p_rate = p_rate
        self.ci_C = None
        self.ci_A = None
        if budget:
            iterations = (self.budget // self.population_size) + 1
            self.step_a = self.a / iterations
        else:
            self.step_a = self.a / self.max_iterations

    def metaheuristic(self):
        """Main Algorithm E-WOA."""
        for i in range(self.population_size):
            # Iterate over each solution
            self.update_A()
            new_population = np.empty_like(self.population)
            emigrants = np.random.choice(self.population_size, self.p_rate, replace=False)

            if i in emigrants:
                # migration search strategy
                best_max = np.max(self.best_global_solution.vector)
                best_min = np.min(self.best_global_solution.vector)
                new_population[i] = np.random.uniform(self.problem.lower_bounds,
                                                      self.problem.upper_bounds,
                                                      self.problem.dimension) - \
                                    np.random.uniform(best_min,
                                                      best_max,
                                                      self.problem.dimension)

            elif self.r() < 0.5:
                # Spiral bubble-net attacking method
                distance = np.abs(self.best_global_solution.vector - self.population[i])
                l_value = np.random.uniform(-1, 1)
                new_population[i] = distance * np.exp(self.b * l_value) * np.cos(2 * np.pi * l_value) + \
                                    self.best_global_solution.vector
            elif np.abs(self.ci_A[i]) >= 0.5:
                # Preferential selecting search strategy
                self.update_C()
                px1, px2 = self.pooling_mechanism(2)
                new_population[i] = self.population[i] + self.ci_A[i] * (self.ci_C[i] * px1 - px2)

            else:
                # Enriched encircling prey search strategy
                self.update_C()
                px3 = self.pooling_mechanism(1)
                dist = np.abs(self.ci_C[i] * self.best_global_solution.vector - px3)
                new_population[i] = self.best_global_solution.vector - self.ci_A[i] * dist

        # Bound Correction
        new_population = self.boundary_correction(new_population)
        new_fitness = np.apply_along_axis(self.problem, 1, new_population)
        mask = np.where(new_fitness < self.evaluation)
        self.population[mask] = new_population[mask]
        self.evaluation[mask] = new_fitness[mask]
        # print(self.best_global_solution.value)

    def update_A(self) -> None:
        # Cauchy Distribution
        self.ci_A = 0.5 + 0.1 * np.tan(np.pi * (self.r(self.population_size) - 0.5))

    def update_C(self) -> None:
        self.ci_C = np.random.uniform(0, 2, self.population_size)

    def pooling_mechanism(self, q):
        idx0 = int((self.population_size - self.kappa * 0.3) + 1)
        selected_worsts = np.random.randint(idx0, self.population_size, q)
        worsts = self.population[selected_worsts]

        best_max = np.max(self.best_global_solution.vector)
        best_min = np.min(self.best_global_solution.vector)

        binary_matrix = np.random.choice([0, 1], size=(q, self.problem.dimension))
        binary_reverted = 1 - binary_matrix
        x_neighbors = np.random.uniform(best_min, best_max, (q, self.problem.dimension))
        current_pool = binary_matrix * x_neighbors + binary_reverted * worsts

        if q == 2:
            return current_pool[0], current_pool[1]
        elif q == 1:
            return current_pool
