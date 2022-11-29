from sko.GA import GA

import networkx as nx
import tensorflow as tf
import random
import os

import yaml

import numpy as np
from openbox import sp
from tqdm import tqdm

random_seed = 123
task_name = 'default_task'
batch_path = './batch_opt'


# Generate a topology according to parameters
def generate_topology(net_size, graph_file, degrees, bufferSizes, bandwidth, bandwidth_weights, schedulingPolicy,
                      bandw_range, buffer_range, buffer_step, schedulingWeight_factor):
    G = nx.Graph()
    nodes = []
    node_degree = degrees.copy()
    random.shuffle(node_degree)
    for i in range(net_size):
        nodes.append(i)
        G.add_node(i)
        # Assign to each node the scheduling Policy
        G.nodes[i]["schedulingPolicy"] = random.choice(schedulingPolicy)
        # G.nodes[i]["schedulingWeights"] = "45, 30, 25"

        avgWeight = 100
        schWeight = []
        for j in range(3):
            schWeight.append(avgWeight + random.randint(-schedulingWeight_factor, schedulingWeight_factor))

        schWeight_sum = sum(schWeight)
        for j in range(3):
            schWeight[j] = int(schWeight[j] / schWeight_sum * 100)
        schWeight_diff = 100 - sum(schWeight)
        schWeight[0] += schWeight_diff
        schWeight.sort(reverse=True)
        # G.nodes[i]["schedulingWeights"] = "45, 30, 25"
        G.nodes[i]["schedulingWeights"] = ', '.join([str(x) for x in schWeight])

        # Assign the buffer size of all the ports of the node
        bs = random.choice(bufferSizes)
        if buffer_range > 0:
            # G.nodes[i]["bufferSizes"] = random.choice(range(max(bs-500,8000),min(bs+1000,64000),500))
            G.nodes[i]["bufferSizes"] = random.choice(
                range(max(bs - buffer_range * buffer_step, 8000), min(bs + (buffer_range + 1) * buffer_step, 64000),
                      buffer_step)
            )
        else:
            G.nodes[i]["bufferSizes"] = bs

    finish = False
    while (True):
        aux_nodes = list(nodes)
        n0 = random.choice(aux_nodes)
        aux_nodes.remove(n0)
        # Remove adjacents nodes (only one link between two nodes)
        for n1 in G[n0]:
            if (n1 in aux_nodes):
                aux_nodes.remove(n1)
        if (len(aux_nodes) == 0):
            # No more links can be added to this node - can not acomplish node_degree for this node
            nodes.remove(n0)
            if (len(nodes) == 1):
                break
            continue
        n1 = random.choice(aux_nodes)
        G.add_edge(n0, n1)
        # Assign the link capacity to the link
        same_bandwidth = random.choices(bandwidth, weights=bandwidth_weights)[0]
        G[n0][n1]["bandwidth"] = random.choice(
            range(same_bandwidth - bandw_range * 1000, same_bandwidth + (bandw_range + 1) * 1000, 1000))

        for n in [n0, n1]:
            node_degree[n] -= 1
            if (node_degree[n] == 0):
                nodes.remove(n)
                if (len(nodes) == 1):
                    finish = True
                    break
        if (finish):
            break
    if (not nx.is_connected(G)):
        G = generate_topology(net_size, graph_file, degrees, bufferSizes, bandwidth, bandwidth_weights,
                              schedulingPolicy, bandw_range, buffer_range, buffer_step, schedulingWeight_factor)
        return G

    nx.write_gml(G, graph_file)

    return (G)


# Functions for route generation
# 用给定下标取值，从全部可能简单路径中取值
def get_bw_through_link_by_index(all_route, bw, index):
    # route: dict(dict(list(int)))
    # bw: np.array([[float]])
    # return list(list(list(float)))
    size = bw.shape[0]

    bw_through_link = [[[] for j in range(size)] for i in range(size)]

    # loop for path
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            route = all_route[i][j][index[i, j]]
            for k in range(1, len(route)):
                # per link's index
                src = route[k - 1]
                dst = route[k]

                bw_through_link[src][dst].append(bw[i][j])

    return bw_through_link


# 计算需求超过供给的差值
def compute_tm_loss(link_bw, bw_through_link):
    loss = 0
    for link, bw in link_bw.items():
        src = link[0]
        dst = link[1]
        use = np.sum(bw_through_link[src][dst])
        route_len = len(bw_through_link[src][dst])
        loss += route_len
        if use < bw:
            continue
        loss += use - bw
    return loss


# given a length threshold, find possible paths
def get_all_route_choices(g, route_len_plus):
    size = g.number_of_nodes()

    all_route_choices = dict()
    for i in range(size):
        all_src_routes = dict()
        for j in range(size):
            if i == j:
                continue
            shortest_routes = []
            # All paths
            route_generator = nx.shortest_simple_paths(g, i, j)
            shortest_len = 0
            specific_len_plus = random.choice(range(0, route_len_plus + 1))
            for k, r in enumerate(route_generator):
                if k == 0:
                    shortest_len = len(r)
                elif len(r) > shortest_len + specific_len_plus:
                    break
                shortest_routes.append(r)
            if len(shortest_routes[-1]) < shortest_len + specific_len_plus:
                specific_len_plus = len(shortest_routes[-1]) - shortest_len
            while len(shortest_routes[0]) < shortest_len + specific_len_plus:
                shortest_routes.pop(0)
            all_src_routes[j] = shortest_routes
        all_route_choices[i] = all_src_routes
    return all_route_choices


# generate a routing using GA
def generate_routing(G, routing_file, avg_bw, route_len_plus):
    g = G
    bw = avg_bw
    size = G.number_of_nodes()
    all_route_choices = get_all_route_choices(g, route_len_plus)

    def obj_func(p):
        # p: size*(size-1) integers
        nonlocal size
        nonlocal all_route_choices
        nonlocal bw
        nonlocal g
        x = p
        k = 0
        index = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                index[i, j] = int(x[k])
                k += 1
        bw_through_link = get_bw_through_link_by_index(all_route_choices, bw, index)
        link_bw = nx.get_edge_attributes(g, name="bandwidth")
        loss = compute_tm_loss(link_bw, bw_through_link)

        return loss

    ub = []
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            ub.append(len(all_route_choices[i][j]) - 0.5)

    ga = GA(
        func=obj_func,
        n_dim=size * (size - 1),
        size_pop=80,
        max_iter=300,
        lb=[0] * (size * (size - 1)),
        ub=ub,
        precision=[int(1)] * (size * (size - 1))
    )
    ga.run()

    with open(routing_file, "w") as r_fd:
        route = ga.best_x.astype("int").tolist()
        cnt = 0
        for src in g:
            for dst in g:
                if (src == dst):
                    continue
                route_index = route[cnt]
                cnt += 1
                path = all_route_choices[src][dst][route_index]
                path_str = ','.join(str(x) for x in path)
                r_fd.write(path_str + "\n")


# Traffic Matrix
def generate_tm(G, max_avg_lbda, traffic_file):
    poisson = "0"
    cbr = "1"
    on_off = "2,5,5"  # time_distribution, avg on_time exp, avg off_time exp
    time_dist = [poisson, cbr, on_off]

    pkt_dist_1 = "0,500,0.1,750,0.16,1000,0.36,1250,0.24,1500,0.14"  # genric pkt size dist, pkt_size 1, prob 1, pkt_size 2, prob 2
    pkt_dist_2 = "0,500,0.22,750,0.05,1000,0.06,1250,0.62,1500,0.05"  # genric pkt size dist, pkt_size 1, prob 1,
    # pkt_size 2, prob 2, pkt_size 3, prob 3
    pkt_dist_3 = "0,500,0.2,750,0.2,1000,0.2,1250,0.2,1500,0.2"
    pkt_size_dist = [pkt_dist_1, pkt_dist_2, pkt_dist_3]
    tos_lst = [0, 1, 2]

    low = random.randint(100, 500)
    high = random.randint(2500, 4000)

    # bandwidth
    bw = np.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=float)

    with open(traffic_file, "w") as tm_fd:
        for src in G:
            for dst in G:
                # avg_bw = random.randint(10,max_avg_lbda)
                avg_bw = random.randint(low, high)
                bw[src, dst] = avg_bw
                td = random.choice(time_dist)
                sd = random.choice(pkt_size_dist)
                tos = random.choice(tos_lst)

                traffic_line = "{},{},{},{},{},{}".format(
                    src, dst, avg_bw, td, sd, tos)
                tm_fd.write(traffic_line + "\n")

    return bw


def generate_graph_file(config: sp.Configuration, random_seed, training_dataset_path, ):
    # all the hyperparameters
    params = config.get_dictionary()

    graph_num = 10
    bandwidth11 = int(params['bandwidth11'])
    bandwidth12 = int(params['bandwidth12'])
    bandwidth13 = int(params['bandwidth13'])
    bandwidth14 = int(params['bandwidth14'])
    bandwidth21 = int(params['bandwidth21'])
    bandwidth22 = int(params['bandwidth22'])
    bandwidth23 = int(params['bandwidth23'])
    bandwidth24 = int(params['bandwidth24'])
    bandwidth31 = int(params['bandwidth31'])
    bandwidth32 = int(params['bandwidth32'])
    bandwidth33 = int(params['bandwidth33'])
    bandwidth34 = int(params['bandwidth34'])
    bandwidth_range = int(params['bandwidth_range'])
    buffer_range = int(params['buffer_range'])
    buffer_step = int(params['buffer_step']) * 250
    route_len_plus = int(params['route_len_plus'])
    schedulingWeight_factor = int(params['schedulingWeight_factor'])

    bandwidth1 = [1000 * bandwidth11, 1000 * bandwidth12, 1000 * bandwidth13, 1000 * bandwidth14]
    bandwidth2 = [1000 * bandwidth21, 1000 * bandwidth22, 1000 * bandwidth23, 1000 * bandwidth24]
    bandwidth3 = [1000 * bandwidth31, 1000 * bandwidth32, 1000 * bandwidth33, 1000 * bandwidth34]
    bandwidth = [
        bandwidth1,
        bandwidth1,
        bandwidth1,
        bandwidth2,
        bandwidth2,
        bandwidth2,
        bandwidth2,
        bandwidth2,
        bandwidth3,
        bandwidth3
    ]

    # fixed parameters
    # useless parameter
    max_avg_lbda = 1000
    # 10 graphs with nodes as follows
    net_size = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # degrees
    degrees = [
        [1, 2, 2, 2, 3, 3, 3, 3, 3, 4],
        [1, 2, 2, 2, 3, 3, 3, 3, 3, 4],
        [2, 3, 3, 3, 3, 3, 3, 3, 4, 5],
        [2, 3, 3, 3, 3, 3, 3, 3, 4, 5],
        [2, 3, 3, 3, 3, 3, 3, 3, 4, 5],
        [2, 3, 3, 4, 4, 4, 4, 4, 5, 5],
        [2, 3, 3, 4, 4, 4, 4, 4, 5, 5],
        [2, 3, 3, 4, 4, 4, 4, 4, 5, 5],
        [3, 3, 4, 4, 4, 4, 5, 5, 5, 6],
        [3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
    ]

    # bandwidth
    bandwidth_weights = [
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299],
        [0.20835841, 0.25996935, 0.29750924, 0.23416299]
    ]

    bufferSizes = [8000, 16000, 32000, 64000]  # as validation set's distribution

    schedulingPolicy = ["FIFO", "SP", "WFQ", "DRR"]  # as validation set's distribution

    # set random seeds
    random_seed = random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # file path
    training_dataset_path = training_dataset_path
    graphs_path = "graphs"
    routings_path = "routings"
    tm_path = "tm"
    simulation_file = os.path.join(training_dataset_path, "simulation.txt")

    # mkdir
    if (os.path.isdir(training_dataset_path)):
        print("Destination path already exists. Files within the directory may be overwritten.")
    else:
        os.makedirs(os.path.join(training_dataset_path, graphs_path))
        os.mkdir(os.path.join(training_dataset_path, routings_path))
        os.mkdir(os.path.join(training_dataset_path, tm_path))

    # data generation
    with open(simulation_file, "w") as fd:
        for index in tqdm(range(0, 10)):
            # Generate graph
            graph_file = os.path.join(graphs_path, "graph_{}.txt".format(index))
            global G
            G = generate_topology(
                net_size[index],
                os.path.join(training_dataset_path, graph_file),
                degrees=degrees[index],
                bufferSizes=bufferSizes,
                bandwidth=bandwidth[index],
                bandwidth_weights=bandwidth_weights[index],
                schedulingPolicy=schedulingPolicy,
                bandw_range=bandwidth_range,
                buffer_range=buffer_range,
                buffer_step=buffer_step,
                schedulingWeight_factor=schedulingWeight_factor
            )

            for i in range(10):
                # Generate TM:
                tm_file = os.path.join(tm_path, "tm_{}_{}.txt".format(index, i))
                bw = generate_tm(G, max_avg_lbda, os.path.join(training_dataset_path, tm_file))

                # Generate routing
                complex_mode = False
                # if index>=9:
                # complex_mode=True
                routing_file = os.path.join(routings_path, "routing_{}_{}.txt".format(index, i))
                generate_routing(G, os.path.join(training_dataset_path, routing_file), bw, route_len_plus)

                sim_line = "{},{},{}\n".format(graph_file, routing_file, tm_file)
                # If dataset has been generated in windows, convert paths into linux format
                fd.write(sim_line.replace("\\", "/"))
    return True


# Get docker cmd
def docker_cmd(training_dataset_path):
    raw_cmd = f"docker run --rm --mount type=bind,src={os.path.join(os.getcwd(), training_dataset_path)},dst=/data bnnupc/netsim:v0.1"
    terminal_cmd = "sudo " + raw_cmd
    return raw_cmd, terminal_cmd


def run_simulator(dataset_name, training_dataset_path):
    dataset_name = dataset_name
    training_dataset_path = training_dataset_path

    # simulator's config
    conf_file = os.path.join(training_dataset_path, "conf.yml")
    conf_parameters = {
        "threads": 4,  # Number of threads to use. This is USELESS
        "dataset_name": dataset_name,  # Name of the dataset. It is created in <training_dataset_path>/results/<name>
        "samples_per_file": 10,  # Number of samples per compressed file
        "rm_prev_results": "n",  # If 'y' is selected and the results folder already exists, the folder is removed.
    }
    with open(conf_file, 'w') as fd:
        yaml.dump(conf_parameters, fd)

    # launch docker
    raw_cmd, terminal_cmd = docker_cmd(training_dataset_path)
    os.system(terminal_cmd)

    return True


def generate_training_data(training_path, config: sp.Configuration):
    # 生成图文件
    print('Generating Graph File..........')
    generate_graph_file(config, random_seed=random_seed, training_dataset_path=training_path)

    # Use sudo docker command to run the OMNet++ simulator
    print('Simulating.....................')
    run_simulator(dataset_name='batchBO', training_dataset_path=training_path)


# Make sure that you have password-free sudo privilege
if __name__ == "__main__":
    from get_config_space import get_best_configspace

    best_config = get_best_configspace().get_default_configuration()
    generate_training_data("./training", best_config)
