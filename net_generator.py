import configuration_model_noloops_noparallel_edges as CM
import networkx as nx
import copy, random, numpy, operator, copy, itertools
import configuration_model_noloops_noparallel_edges as CM

def network_construct(net_size, k_min, k_max, *arg):
    degree_distribution_type = arg[0]

    origin_net = nx.Graph()
    z = list()
    is_degree_sequence_valid = False

    if degree_distribution_type == "power law":
        sf_exponent = arg[1]
        while not is_degree_sequence_valid:
            z = list()
            while len(z) < net_size:
                newsample = round(random.paretovariate(sf_exponent - 1))
                if newsample <= k_max and newsample >= k_min:
                    z.extend([newsample])
      
            is_degree_sequence_valid = sum(z) % 2 == 0

        connected_test = False

        print("Got valid z")
        while not connected_test:
            origin_net = CM.configuration_model(z)
            connected_test = nx.is_connected(origin_net)
            print("Connected?:", connected_test)


    elif degree_distribution_type == "BA model": # Barabasi-Albert model
        origin_net = nx.barabasi_albert_graph(net_size, m_links)

    elif degree_distribution_type == "Poisson":
        ave_degree = arg[1]
        while not is_degree_sequence_valid:
            z = list()
            while len(z) < net_size:
                newsample = round(numpy.random.poisson(ave_degree))
                if newsample <= k_max and newsample >= k_min:
                    z.extend([newsample])

            # is_degree_sequence_valid = nx.is_valid_degree_sequence(z)
            is_degree_sequence_valid = sum(z) % 2 == 0
        connected_test = False
        while not connected_test:
            origin_net = CM.configuration_model(z)
            connected_test = nx.is_connected(origin_net)

    # Remove parallel edges:
    origin_net = nx.Graph(origin_net)
    # Remove self loops:
    origin_net.remove_edges_from(nx.selfloop_edges(origin_net))
    return origin_net
