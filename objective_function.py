from training_data_generation import generate_training_data
import tensorflow as tf
import random
import os
import time
from RouteNet_Fermi import main
from RouteNet_Fermi import evaluate
from openbox import sp

from get_config_space import get_best_configspace

random_seed = 123
task_name = 'default_task'
batch_tmp_path = './batch_opt'


def batch_objective_function(config: sp.Configuration):
    global random_seed, task_name, batch_tmp_path

    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)

    # generate a directory's name randomly
    this_iter = random.random()
    working_dir = batch_tmp_path + '/batch_iter_' + str(this_iter) + 'tmp'
    training_path = working_dir + '/training'
    checkpoint_path = working_dir + '/modelCheckpoints'

    print("Training data's tmp directory: " + training_path)

    generate_training_data(training_path=training_path, config=config)

    # Train RouteNet-Fermi model
    main(training_path, final_evaluation=False, ckpt_dir=checkpoint_path)

    # Find the best checkpoint
    model_file = os.listdir(checkpoint_path)
    best_point = " "
    best_loss = 100
    for modelCheckpoint in model_file:
        if "index" in modelCheckpoint:
            try:
                loss = float(modelCheckpoint[3:8])
            except:
                loss = float(modelCheckpoint[3:7])
            if loss <= best_loss:
                best_loss = loss
                best_point = modelCheckpoint[:-6]
    best_model = checkpoint_path + '/' + best_point
    # loss = evaluate(best_model)
    # get model's loss，
    # RouteNet-Fermi's 'evaluate' API doesn't return any loss value, we didn't modify it,
    # but use the best 20-step validation loss('best_loss') as objective function's return value


    # save the model

    store_path = './batch_history/' + "batch_" + time.strftime("%Y_%m_%d_%H_%M_%S",
                                                               time.localtime()) + "_step20_val_loss_" + str(best_loss)
    try:
        os.mkdir('./batch_history')
    except:
        print("Path Exists")
    os.system('sudo mv ' + working_dir + ' ' + store_path)
    print("Result's directory: " + store_path)

    f = open(store_path + '/config.txt', 'w', encoding='utf-8')
    f.write(str(config))
    f.close()

    return float(best_loss)


if __name__ == "__main__":
    best_config = get_best_configspace().get_default_configuration()
    val_loss = batch_objective_function(config=best_config)
    print(val_loss)
