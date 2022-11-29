# Tutorial


## Structure
- ```RouteNet_Fermi and validation_dataset```: provided by the challenge's organizer, we didn't modify them.
- ```requirements```: exported by conda and pip.
- ```training_data_generation.py```: using this module, you can generate our approach's training data.
- ```objective_function.py```: objective function for Bayesian Optimization(BO) that can be evaluated in parallel.

- ```get_config_sapce.py```: includes our preset hyperparameter's search space and the best hyperparameters we found.
- ```opt_batch.py```: Parallel BO for hyperparameter optimization.
- ```tutorial.md```




## Set Up
1. Our experiment is on Ubuntu 18.04, but it should also perform well on other linux systems.

2. Please ensure that the user running this program has ```sudo``` privilege without password.

3. Please ensure that the python interpreter's environment matches the requirements and that you can run the ```sudo docker``` command properly.


## Training Data Generation
You can generate our best submission's training data directly with the following command:
```
python training_data_generation.py
```
After generation, the training data will be saved in ```./training``` folder.


You can use RouteNet-Fermi's API to train the model:
```
main("./training", final_evaluation = False, ckpt_dir="./modelCheckpoints")
```
Since we reset the random seed to 123 in training data generation, the model's validation loss may differ a little from our solution if it is trained seperately.




## Model Training and Evaluation
We provide a function to generate the training data and the trained model simultaneously.

```
python objective_function.py
```
it automatically calls ```training_data_generation.py``` and trains the model using RouteNet-Fermi's API.
The  ```.\batch_opt``` folder will be used to temporarily store the data during the training. When the training is complete, the ```objective_function``` will evaluate the validation loss of the model and move all the temporary files to the ```.\batch_history``` folder, where you can find the complete training data and all the model checkpoints.







## Parallel Bayesian Optimization
If you try to lauch the whole hyperparameter search process,
you can start Parallel Bayesian Optimization as follows:

```
python opt_batch.py
```
You may need to change some parameters in PBO, such as ```batch_size``` and ```max_runs```. The entire search process may take several hours to days according to preset parameters. All the historical data will be saved in ```./batch_history``` folder.






## Config Space
```get_config_space.py``` contains our preset search space for Bayesian Optimization and the best hyperparameters we've found. You don't need to make any changes.



Also, we include a set of alternative parameters that may be better than our current optimal submission results, you can use it to generate training data as follows:
```
from get_config_space import get_alternative_configspace
from training_data_generation import generate_training_data
generate_training_data('./training', get_alternative_configspace().get_default_configuration())

```
