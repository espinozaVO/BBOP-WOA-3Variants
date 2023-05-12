from metah import *
import multiprocessing as mp
from experiment import Experiment
import numpy as np
import time
import cocoex



if __name__ == '__main__':
    time0 = time.time()
    seed = 42
    ee = Experiment(DE, seed) # Write name of metaheuristic  E_WOA, WOA or SDWOA
    ee.run()
    print("*** Full is  done in %s ***"
          % (cocoex.utilities.ascetime(time.time() - time0)))
   
