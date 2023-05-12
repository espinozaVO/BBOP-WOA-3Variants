from metah.abstractMetaheuristic import AbstractMetaheuristic as absMeta
from metah.Problems import Problem
import numpy as np


class SDWOA(absMeta):
    """modified Symbiotic organism search - Differential Evolution - WOA.

    The modified Symbiotic Organism Search - Differential Evolution - Whale Optimization Algorithm 
    (m-SDWOA) is a hybrid metaheuristic algorithm that partially combines three different algorithms.
    
    The Symbiotic Organism Search (SOS) algorithm is based on the mutualistic symbiotic relationship 
    between different organisms, Differential Evolution (DE), and Whale Optimization Algorithm (WOA) 
    
    Attributes:
    -----------
        ci_A : float
            Coefficient A to weight the distance between X*(t) and X_i(t).
        ci_C : float
            Coefficient C to weight X*(t).
    
    """
    INFO = {
        'author': 'Chakraborty',
        'year': 2023,
        'name': 'Hybrid Whale Optimization Algorithm',
        'short': 'm-SDWOA'
    }

    def __init__(self, population_size: int, max_iterations: int = None, budget: int = None,
                 problem: Problem = None, seed=None, a1: float = 2, b_value: float = 1, gamma: float = 1,
                 t: float = 0, fm: float = 0.5, pc: float =0.9) -> None:
        """
        Parameters:
        -----------
            population_size : int
                Number of solution evaluated together every iteration.
            max_iterations : int
                Maximum number of iteration.
            buget : int
                Maximun number objetive evaluation
            problem : Object
                The problem to optimize.
            seed : Int
                Number to initialize RNG
            a1 : float
                Linearly decreased value
            b_value : float
                A shape constant
            gamma: float
                A linear decresing parameter to change explataition and
                exploration phase.
            t: float
                To determinate size step on each iteration.
            fm : float
               Factor of mutation.This parameter is used to control the magnitude of the
               perturbations applied to the candidate solutions during the mutation phase.
               This typically ranges from 0 to and the default value is 0.9, which means that
               the perturbations can be as large as 90% of the difference between the upper
               and lower bounds of the variables.
           pc : float
               Probability of crossover. A float value that represents the probability of crossover.
               This parameter is used to control the balance between exploration and exploitation
               in the search process. The default value is 0.7, which means that there is a 70%
               chance of performing crossover for each candidate solution.
        """
        super().__init__(population_size, max_iterations, budget, problem, seed)
        self.a1 = a1
        self.a2 = -2
        self.b = b_value
        self.gamma = gamma
        self.ci_C = None
        self.ci_A = None
        self.t = t
        self.fm = fm
        self.pc = pc
        if budget:
            self.max_iterations = (self.budget // self.population_size) + 1
        self.step = self.t / self.max_iterations

    def metaheuristic(self):
        """Main Algorithm WOA."""
        for i in range(self.population_size):
            # Iterate over each solution
            if self.gamma > self.r():
                if self.r() < 0.5:
                    # Commensalism from SOS
                    indices = self.get_partner_index(i, size=2)
                    min_id, max_id = indices[np.argsort(self.evaluation[indices])]
                    partner_b = self.population[min_id]
                    partner_c = self.population[max_id]
                    mean_vector = np.mean([self.population[i], partner_c], axis=0)

                    new_solution1 = self.population[i] + self.r() * (partner_b - mean_vector * (1 + self.t * self.step))
                    new_solution1 = self.boundary_correction(new_solution1)
                    new_fit1 = self.problem(new_solution1)

                    new_solution2 = self.population[max_id] + self.r() * (partner_b - mean_vector * (2 - self.t * self.step))
                    new_solution2 = self.boundary_correction(new_solution2)
                    new_fit2 = self.problem(new_solution2)

                    if new_fit1 < self.evaluation[i]:
                        self.population[i] = new_solution1
                        self.evaluation[i] = new_fit1
                    if new_fit2 < self.evaluation[max_id]:
                        self.population[max_id] = new_solution2
                        self.evaluation[max_id] = new_fit2
                else:
                    # DE / rand / 1 / bin
                    indices = self.get_partner_index(i, 3)
                    a, b, c = self.population[indices]
                    mask = self.r(self.problem.dimension) < self.pc
                    trial_vector = np.where(mask, a + self.fm * (b - c), self.population[i])
                    trial_vector = self.boundary_correction(trial_vector)
                    trial_fitness = self.problem(trial_vector)

                    if trial_fitness < self.evaluation[i]:
                        self.evaluation[i] = trial_fitness
                        self.population[i] = trial_vector

            else:
                if self.r() < 0.5:
                    # Encircling prey WOA
                    self.update_A()
                    self.update_C()
                    distance = np.abs(self.ci_C[i] * self.best_global_solution.vector - self.population[i])
                    new_solution = self.best_global_solution.vector - self.ci_A[i] * distance

                else:
                    # Attacking WOA
                    distance = np.abs(self.best_global_solution.vector - self.population[i])
                    l_value = np.random.uniform(self.a2, 1)
                    new_solution = distance * np.exp(self.b * l_value) * np.cos(2 * np.pi * l_value) + \
                                         self.best_global_solution.vector

                new_solution = self.boundary_correction(new_solution)
                new_fit = self.problem(new_solution)
                if new_fit < self.evaluation[i]:
                    self.evaluation[i] = new_fit
                    self.population[i] = new_solution

        self.t += 1
        self.a1 -= 2 * self.step
        self.a2 += self.step
        self.gamma -= self.step

    def update_A(self) -> None:
        self.ci_A = np.random.uniform(-self.a, self.a, self.population_size)

    def update_C(self) -> None:
        self.ci_C = np.random.uniform(0, 2, self.population_size)
