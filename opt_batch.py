import time

from objective_function import batch_objective_function
from get_config_space import get_search_configspace

import pickle
from matplotlib import pyplot as plt

from openbox import ParallelOptimizer

if __name__ == "__main__":
    '''
    Parallel BO
    
    Parallel optimization on local machine, it will take a long time according to parameters, e.g. batch_size and max_runs
    In our solution, batch_size = 60 and max_runs = 150, which indicates that there are 60 processes
    evaluating the objective function in parallel, and the total numbers of observations is 150.
    
    
    '''
    opt = ParallelOptimizer(
        objective_function=batch_objective_function,
        config_space=get_search_configspace(),
        parallel_strategy='async',
        batch_size=60,
        batch_strategy='default',
        num_objs=1,
        num_constraints=0,
        max_runs=150,
        surrogate_type='gp',
        time_limit_per_trial=72000,
        task_id='parallel_async_optimization',
    )
    history = opt.run()

    print(history)
    plt.show()
    history.plot_convergence()
    plt.savefig('batch_optimization' + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.png')

    # Store optimization history
    f = open('batch_optimization' + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'wb')
    pickle.dump(history, f, 0)
    f.close()

