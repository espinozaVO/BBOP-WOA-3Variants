import cocoex  # experimentation module
from cocoex.utilities import ProblemNonAnytime
import os  # to show post-processed results in the browser
import numpy as np
import time  # output some timings per evaluation
import webbrowser
try:
    import cocopp  # post-processing module
except:
    pass
from collections import defaultdict


class Experiment:

    def __init__(self, class_solver, seed):
        self.solver = class_solver
        self.seed = seed
        self.suite_name = "bbob"
        self.suite_filters = ""  # "dimensions : 2, instance_indices: 1"
        self.suite = cocoex.Suite(self.suite_name, "", self.suite_filters)
        self.output_folder = '%s' % (self.solver.INFO["short"])
        # %s_s%d_f' % (self.solver.INFO["short"], self.seed)
        self.observer = cocoex.Observer(self.suite_name, "result_folder: " + self.output_folder + " " +
                                        "algorithm_name: " + self.solver.INFO["name"])
        self.minimal_print = cocoex.utilities.MiniPrint()
        self.suite_len = len(self.suite)
        self.timings = defaultdict(list)  # key is the dimension

    def run(self):
        for problem in self.suite:
            time1 = time.time()
            size = 100
            budget = (10 ** 4) * problem.dimension
            problem.observe_with(self.observer)  # generate the data for cocopp post-processing
            current_solver = self.solver(population_size=size, budget=budget, problem=problem, seed=self.seed)
            x, y = current_solver.run()
            self.timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                          if problem.evaluations else 0)
            # print(problem.evaluations)
            self.minimal_print(problem, final=problem.index == len(self.suite) - 1)

        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (self.minimal_print.stime, sorted(self.timings)[-1], np.median(self.timings[sorted(self.timings)[-1]])))
        print("Timing summary:\n"
              "  dimension  median seconds/evaluations\n"
              "  -------------------------------------")
        for dimension in sorted(self.timings):
            print("    %3d       %.1e" % (dimension, np.median(self.timings[dimension])))
        print("  -------------------------------------")

    @staticmethod
    def default_budget_list(max_budget=10, num=50):
        """Produces a budget list with at most `num` different increasing budgets
        within [1, `max_budget`] that are equally spaced in the logarithmic space.
        """
        return np.unique(np.logspace(10, np.log10(max_budget) + 10, num=num).astype(int))

    def post_processing(self):
        cocopp.main(self.observer.result_folder)  # re-run folders look like "...-001" etc
        webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

    def free(self):
        self.suite.free()

