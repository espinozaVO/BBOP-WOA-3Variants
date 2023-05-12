from abc import ABC, abstractmethod
from metah.Problems import Problem, Solution
import numpy as np


class AbstractMetaheuristic(ABC):
    """Abstract Class for Population-Based Metaheuristic.

    Attributes:
    -----------
        best_global_solution : Object
            A Solution-Object with the best vector and value find
        population : 2D-array
            A matrix of n solutions in rows and d variables in columns
        fitness : 1D-array
            An array of fitness evaluations
        population_size : int
            Number of solution evaluated together every iteration.
        max_iterations : int
            Maximum number of iteration.
        problem : Object
            The problem to optimize.
        seed : Int
            Number to initialize RNG

    Methods:
    --------
        run
            Execute metaheuristic
        metaheuristic
            A particular population-based metaheuristic.
            This method will be rewritten by the children classes.
        initialize_population
            Encapsulate next 3 methods.
        generate_random_population
            Create a random initial population.
        evaluate_population
            Compute fitness of each solution in population.
        get_best_global_solution
            Find the minimum fitness value. Save minimum value, solution and index.
        boundary_correction
            Set the values of

    """

    def __init__(self, population_size: int = None, max_iterations: int = None, budget: int = None,
                 problem: Problem = None, seed=None) -> None:
        """
        Parameters
        ----------
        population_size : int
            The number of candidate solutions in the population. The actual number of
            candidate solutions in the population is half of the given population_size,
            because new solutions are generated in two phases: the employer phase and
            the onlooker phase.
        max_iterations : int
            The maximum number of iterations to run the algorithm.
        problem : Problem
            The optimization problem to be solved. The problem should have the following
            attributes: lower_bounds, upper_bounds, dimension, and __call__.
            - lower_bounds: a 1D numpy array containing the lower bounds of the variables.
            - upper_bounds: a 1D numpy array containing the upper bounds of the variables.
            - dimension: an integer specifying the number of variables.
            - __call__(x): the objective function. A method that takes a 1D numpy array of
                           shape (dimension,) as input and returns a scalar value as output.
        seed : int, optional
            The random seed to use for initializing the population. If not provided, a
            random seed will be used.
        """
        # Declare Local Void Variables
        self.best_global_solution = Solution()
        self.worst_global_solution = Solution()
        self.population = None
        self.evaluation = None
        self.seed = seed
        self.r = np.random.random  # A shortname to random np function
        np.random.seed(seed=self.seed)
        # Another way to initialize RNG.
        # np.random.default_rng(seed=self.seed)

        # Set Parameters
        self.population_size = population_size
        if (not max_iterations) == (not budget):
            raise ValueError('Expected either :max_iterations xor :budget')
        else:
            self.max_iterations = max_iterations
            self.budget = budget
        self.problem = problem

    def run(self):
        """ Execute metaheuristic """
        self.initialize_population()
        if self.budget:
            while self.problem.evaluations < self.budget:
                if self.problem.final_target_hit:
                    break
                self.metaheuristic()
                self.get_best_global_solution()
        else:
            for _ in range(self.max_iterations):
                if self.problem.final_target_hit:
                    break
                self.metaheuristic()
                self.get_best_global_solution()

        return self.best_global_solution.vector, self.best_global_solution.value

    def initialize_population(self):
        """ Initialize population.

        Encapsulate 3 methods needed to give the initial conditions in
        a population-based metaheuristic
            generate_random_population
            evaluate_population
            get_best_global_solution
        """
        self.generate_random_population()
        self.evaluate_population()
        self.get_best_global_solution()

    @abstractmethod
    def metaheuristic(self):
        """ A particular population-based metaheuristic.

        This method will be rewritten by the children classes.
        """
        pass

    def generate_random_population(self):
        """ Create a random initial population."""
        self.population = np.random.uniform(self.problem.lower_bounds,
                                            self.problem.upper_bounds,
                                            (self.population_size, self.problem.dimension))

    def evaluate_population(self):
        """ Compute fitness of each solution in population."""
        self.evaluation = np.apply_along_axis(self.problem, 1, self.population)

    def get_best_global_solution(self):
        """ Find the minimum fitness value. Save minimum value, solution and index."""
        if self.best_global_solution.value is None \
                or self.best_global_solution.value > np.min(self.evaluation):
            self.best_global_solution.value = np.min(self.evaluation)
            self.best_global_solution.index = np.argmin(self.evaluation)
            self.best_global_solution.vector = self.population[self.best_global_solution.index]

    def get_worst_global_solution(self):
        """ Find the minimum fitness value. Save minimum value, solution and index."""
        if self.worst_global_solution.value is None \
                or self.best_global_solution.value > np.min(self.evaluation):
            self.worst_global_solution.value = np.max(self.evaluation)
            self.worst_global_solution.index = np.argmax(self.evaluation)
            self.worst_global_solution.vector = self.population[self.worst_global_solution.index]

    def boundary_correction(self, value):
        if isinstance(value, int):
            self.population[value] = np.clip(self.population[value], self.problem.lower_bounds, self.problem.upper_bounds)
        elif isinstance(value, np.ndarray):
            return np.clip(value, self.problem.lower_bounds, self.problem.upper_bounds)

    def get_partner_index(self, index, size=None):
        indexes_array = np.arange(self.population_size)
        indexes_array = indexes_array[indexes_array != index]
        selected_indexes = np.random.choice(indexes_array, size=size)
        return selected_indexes
