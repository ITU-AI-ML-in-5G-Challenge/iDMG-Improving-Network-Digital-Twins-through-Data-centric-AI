from openbox import sp


def get_search_configspace():
    # Get hyperparameter's configspace for Bayesian Optimization

    space = sp.Space()
    bandwidth11 = sp.Int('bandwidth11', 18, 30, default_value=20)
    bandwidth12 = sp.Int('bandwidth12', 30, 40, default_value=35)
    bandwidth13 = sp.Int('bandwidth13', 46, 56, default_value=51)
    bandwidth14 = sp.Int('bandwidth14', 84, 94, default_value=89)
    bandwidth21 = sp.Int('bandwidth21', 15, 25, default_value=20)
    bandwidth22 = sp.Int('bandwidth22', 30, 40, default_value=35)
    bandwidth23 = sp.Int('bandwidth23', 46, 56, default_value=51)
    bandwidth24 = sp.Int('bandwidth24', 84, 94, default_value=89)
    bandwidth31 = sp.Int('bandwidth31', 15, 25, default_value=20)
    bandwidth32 = sp.Int('bandwidth32', 30, 40, default_value=35)
    bandwidth33 = sp.Int('bandwidth33', 46, 56, default_value=51)
    bandwidth34 = sp.Int('bandwidth34', 84, 94, default_value=89)
    bandwidth_range = sp.Int('bandwidth_range', 1, 3, default_value=1)
    buffer_range = sp.Int('buffer_range', 0, 2, default_value=0)
    buffer_step = sp.Int('buffer_step', 1, 4, default_value=1)
    route_len_plus = sp.Int('route_len_plus', 0, 6, default_value=3)
    schedulingWeight_factor = sp.Int('schedulingWeight_factor', 0, 75, default_value=25)

    space.add_variables([bandwidth11, bandwidth12, bandwidth13, bandwidth14,
                         bandwidth21, bandwidth22, bandwidth23, bandwidth24,
                         bandwidth31, bandwidth32, bandwidth33, bandwidth34,
                         bandwidth_range, buffer_range, buffer_step, route_len_plus, schedulingWeight_factor])
    return space


def get_best_configspace():
    # The configspace itself is as same as get_search_configspace, but the default_value has been set to the optimal value we found
    space = sp.Space()
    bandwidth11 = sp.Int('bandwidth11', 18, 30, default_value=24)
    bandwidth12 = sp.Int('bandwidth12', 30, 40, default_value=38)
    bandwidth13 = sp.Int('bandwidth13', 46, 56, default_value=48)
    bandwidth14 = sp.Int('bandwidth14', 84, 94, default_value=91)
    bandwidth21 = sp.Int('bandwidth21', 15, 25, default_value=17)
    bandwidth22 = sp.Int('bandwidth22', 30, 40, default_value=30)
    bandwidth23 = sp.Int('bandwidth23', 46, 56, default_value=46)
    bandwidth24 = sp.Int('bandwidth24', 84, 94, default_value=94)
    bandwidth31 = sp.Int('bandwidth31', 15, 25, default_value=17)
    bandwidth32 = sp.Int('bandwidth32', 30, 40, default_value=36)
    bandwidth33 = sp.Int('bandwidth33', 46, 56, default_value=55)
    bandwidth34 = sp.Int('bandwidth34', 84, 94, default_value=93)
    bandwidth_range = sp.Int('bandwidth_range', 1, 3, default_value=1)
    buffer_range = sp.Int('buffer_range', 0, 2, default_value=2)
    buffer_step = sp.Int('buffer_step', 1, 4, default_value=4)
    route_len_plus = sp.Int('route_len_plus', 0, 6, default_value=4)
    schedulingWeight_factor = sp.Int('schedulingWeight_factor', 0, 75, default_value=66)

    space.add_variables([bandwidth11, bandwidth12, bandwidth13, bandwidth14,
                         bandwidth21, bandwidth22, bandwidth23, bandwidth24,
                         bandwidth31, bandwidth32, bandwidth33, bandwidth34,
                         bandwidth_range, buffer_range, buffer_step, route_len_plus, schedulingWeight_factor])
    return space


def get_alternative_configspace():
    space = sp.Space()
    bandwidth11 = sp.Int('bandwidth11', 18, 30, default_value=28)
    bandwidth12 = sp.Int('bandwidth12', 30, 40, default_value=31)
    bandwidth13 = sp.Int('bandwidth13', 46, 56, default_value=49)
    bandwidth14 = sp.Int('bandwidth14', 84, 94, default_value=87)
    bandwidth21 = sp.Int('bandwidth21', 15, 25, default_value=22)
    bandwidth22 = sp.Int('bandwidth22', 30, 40, default_value=35)
    bandwidth23 = sp.Int('bandwidth23', 46, 56, default_value=50)
    bandwidth24 = sp.Int('bandwidth24', 84, 94, default_value=87)
    bandwidth31 = sp.Int('bandwidth31', 15, 25, default_value=24)
    bandwidth32 = sp.Int('bandwidth32', 30, 40, default_value=33)
    bandwidth33 = sp.Int('bandwidth33', 46, 56, default_value=48)
    bandwidth34 = sp.Int('bandwidth34', 84, 94, default_value=92)
    bandwidth_range = sp.Int('bandwidth_range', 1, 3, default_value=1)
    buffer_range = sp.Int('buffer_range', 0, 2, default_value=0)
    buffer_step = sp.Int('buffer_step', 1, 4, default_value=4)
    route_len_plus = sp.Int('route_len_plus', 0, 6, default_value=3)
    schedulingWeight_factor = sp.Int('schedulingWeight_factor', 0, 75, default_value=55)

    space.add_variables([bandwidth11, bandwidth12, bandwidth13, bandwidth14,
                         bandwidth21, bandwidth22, bandwidth23, bandwidth24,
                         bandwidth31, bandwidth32, bandwidth33, bandwidth34,
                         bandwidth_range, buffer_range, buffer_step, route_len_plus, schedulingWeight_factor])
    return space
