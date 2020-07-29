import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
import time, sys
import net_generator
import pickle

dec_places = 6
state_epsilon = 0.0001


def eta_test(g1, g2):
    """ Check if g1 and g2 have the same no. of links No_links and eta = No. same links / No_links
    """
    #print("testing no. of links on two layers must be equal:", g1.number_of_edges(), g2.number_of_edges())
    no_same_links = len(list([e for e in g1.edges() if e in g2.edges()]))
    return (no_same_links / g1.number_of_edges())

def generate_G2(g1, n):
    g2 = copy.deepcopy(g1)
    if n == 1: return g2

    edges_of_g1 = list(g1.edges())
    random.shuffle(edges_of_g1)

    no_links = g1.number_of_edges()
    no_iterations = round(no_links * (1 - n) / 2)
    rewired_links = []

    iter = 0
    for AB in edges_of_g1:
        remain_edges = [i for i in edges_of_g1 if i != AB]

        for CD in remain_edges:
            if g2.has_edge(*AB) and g2.has_edge(*CD) \
                    and (not (g1.has_edge(AB[0], CD[0]) or g2.has_edge(AB[0], CD[0]))) \
                    and (not (g1.has_edge(AB[1], CD[1]) or g2.has_edge(AB[1], CD[1]))):
                g2.remove_edges_from([AB, CD])
                g2.add_edges_from([(AB[0], CD[0]), (AB[1], CD[1])])
                iter += 1

                for i in [AB, CD, (AB[0], CD[0]), (AB[1], CD[1])]:
                    rewired_links += [i]
                break

        if iter == no_iterations: break

    return [g2, rewired_links]

def generate_G2_G3(g1, eta_g1g2, eta_g2g3, eta_g3g1, max_tries = 5):
    """ Given g1, generate g2 and g3
    :param
    :return
    """
    count_try = 1
    while True:
        g2 = generate_G2(g1, eta_g1g2)[0]
        g3 = generate_G2(g1, eta_g3g1)[0]
        eta_g2g3_ini = eta_test(g2, g3)
        print('eta_g2g3 starting at: ', eta_g2g3_ini)

        if eta_g2g3_ini <= eta_g2g3:
            edges_of_g2 = list(g2.edges())
            edges_of_g3 = list(g3.edges())
            intersect_g2g3_ini = set(edges_of_g3).intersection(set(edges_of_g2)) #BnC
            diff_g2g3_ini = set(edges_of_g3) - intersect_g2g3_ini #C-B
            no_iterations = round(len(intersect_g2g3_ini) * (eta_g2g3 - eta_g2g3_ini) / eta_g2g3_ini / 2)

            iter = 0
            # rewiring g3:
            print('rewiring G3...')
            for AB in diff_g2g3_ini:
                # for AB in edges_of_g3:
                remain_edges = [i for i in diff_g2g3_ini if i != AB]

                for CD in remain_edges:
                    if g3.has_edge(*AB) and g3.has_edge(*CD): #Take edges from C itself, map into BnC
                        if not (g3.has_edge(AB[0], CD[0]) or g3.has_edge(AB[1], CD[1])): #C does not have mapping
                            if ((g2.has_edge(AB[0], CD[0]) and g2.has_edge(AB[1], CD[1]))) \
                                    and ((not (g1.has_edge(*AB) or g1.has_edge(*CD) or g1.has_edge(AB[0], CD[0]) or g1.has_edge(AB[1], CD[1]))) or \
                                         (g1.has_edge(*AB) and g1.has_edge(*CD) and g1.has_edge(AB[0], CD[0]) and g1.has_edge(AB[1], CD[1])) or \
                                         (g1.has_edge(*AB) and g1.has_edge(AB[0], CD[0]) and (not g1.has_edge(*CD)) and (not g1.has_edge(AB[1], CD[1]))) or \
                                         (g1.has_edge(*CD) and g1.has_edge(AB[1], CD[1]) and (not g1.has_edge(*AB)) and (not g1.has_edge(AB[0], CD[0])))):#A has half(i.e cancels out)
                                g3.remove_edges_from([AB, CD])
                                g3.add_edges_from([(AB[0], CD[0]), (AB[1], CD[1])])
                                iter += 1
                                print('iter ', iter, 'out of', no_iterations)
                                break
                            #line81 B has mapping (increases BnC)
                            #line82 A does not have all (mapping or edges)(No effect on AnC)
                            #line83 A has all (No effect on AnC as removing dec. AnC and adding inc. it back)
                            #line84 A has half(i.e cancels out)

                        elif not (g3.has_edge(AB[0], CD[1]) or g3.has_edge(AB[1], CD[0])): #if C has previous mapping, try other combination for mapping
                            if ((g2.has_edge(AB[0], CD[1]) and g2.has_edge(AB[1], CD[0]))) \
                                    and ((not (g1.has_edge(*AB) or g1.has_edge(*CD) or g1.has_edge(AB[0], CD[1]) or g1.has_edge(AB[1], CD[0]))) or \
                                         (g1.has_edge(*AB) and g1.has_edge(*CD) and g1.has_edge(AB[0], CD[1]) and g1.has_edge(AB[1], CD[0])) or \
                                         (g1.has_edge(*AB) and g1.has_edge(AB[1], CD[0]) and (not g1.has_edge(*CD)) and (not g1.has_edge(AB[0], CD[1]))) or \
                                         (g1.has_edge(*AB) and g1.has_edge(AB[1], CD[0]) and (not g1.has_edge(*CD)) and (not g1.has_edge(AB[0], CD[1])))):#A has half(i.e cancels out)
                                g3.remove_edges_from([AB, CD])
                                g3.add_edges_from([(AB[0], CD[1]), (AB[1], CD[0])])
                                iter += 1
                                print('iter ', iter, 'out of', no_iterations)
                                break
                            #line97 B has mapping (increases BnC)
                            #line98 A does not have all (mapping or edges)(No effect on AnC)
                            #line99 A has all (No effect on AnC)
                            #line100 A has half(i.e cancels out)

                if iter >= no_iterations: return g2, g3

            # Rewiring g2:
            print("rewiring G2...")
            edges_of_g2 = list(g2.edges())
            edges_of_g3 = list(g3.edges())
            intersect_g2g3_ini = set(edges_of_g3).intersection(set(edges_of_g2))
            diff_g3g2_ini = set(edges_of_g2) - intersect_g2g3_ini

            for AB in diff_g3g2_ini:
                remain_edges = [i for i in diff_g3g2_ini if i != AB]

                for CD in remain_edges:
                    if g2.has_edge(*AB) and g2.has_edge(*CD):
                        if not (g2.has_edge(AB[0], CD[0]) or g2.has_edge(AB[1], CD[1])):
                            if ((g3.has_edge(AB[0], CD[0]) and g3.has_edge(AB[1], CD[1]))) \
                                    and ((not (g1.has_edge(*AB) or g1.has_edge(*CD) or g1.has_edge(AB[0], CD[0]) or g1.has_edge(AB[1], CD[1]))) or \
                                         (g1.has_edge(*AB) and g1.has_edge(*CD) and g1.has_edge(AB[0], CD[0]) and g1.has_edge(AB[1], CD[1])) or \
                                         (g1.has_edge(*AB) and g1.has_edge(AB[0], CD[0]) and (not g1.has_edge(*CD)) and (not g1.has_edge(AB[1], CD[1]))) or \
                                         (g1.has_edge(*CD) and g1.has_edge(AB[1], CD[1]) and (not g1.has_edge(*AB)) and (not g1.has_edge(AB[0], CD[0])))):
                                g2.remove_edges_from([AB, CD])
                                g2.add_edges_from([(AB[0], CD[0]), (AB[1], CD[1])])
                                iter += 1
                                print('iter ', iter, 'out of', no_iterations)
                                break

                        elif not (g2.has_edge(AB[0], CD[1]) or g2.has_edge(AB[1], CD[0])):
                            if ((g3.has_edge(AB[0], CD[1]) and g3.has_edge(AB[1], CD[0]))) \
                                    and ((not (g1.has_edge(*AB) or g1.has_edge(*CD) or g1.has_edge(AB[0], CD[1]) or g1.has_edge(AB[1], CD[0]))) or \
                                         (g1.has_edge(*AB) and g1.has_edge(*CD) and g1.has_edge(AB[0], CD[1]) and g1.has_edge(AB[1], CD[0])) or \
                                         (g1.has_edge(*AB) and g1.has_edge(AB[1], CD[0]) and (not g1.has_edge(*CD)) and (not g1.has_edge(AB[0], CD[1]))) or \
                                         (g1.has_edge(*AB) and g1.has_edge(AB[1], CD[0]) and (not g1.has_edge(*CD)) and (not g1.has_edge(AB[0], CD[1])))):
                                g2.remove_edges_from([AB, CD])
                                g2.add_edges_from([(AB[0], CD[1]), (AB[1], CD[0])])
                                iter += 1
                                print('iter ', iter, 'out of', no_iterations)
                                break

                if iter >= no_iterations: return g2, g3

        else:  # we want etaAB = etaBC > etaCA so  eta_g2g3_ini <= eta_g2g3 in most, if not all, cases, so no need to mind this branch for now.
            sys.exit("Oops...it happens??")

        print('count try', count_try)
        count_try += 1
        if count_try > max_tries:
            sys.exit("Cannot complete the task. Increase max_tries or input other argument values")


def generate_G3(g2, g1, nBC, nAC, nAB):
    g3 = copy.deepcopy(g2)
    if nBC == 1: return g3

    edges_of_g2 = list(g2.edges())
    edges_of_g1 = list(g1.edges())

    no_links = g2.number_of_edges()
    no_iterations = round(no_links * (1 - nBC) / 2)
    no_iterations2 = round(no_links * (nAB - nAC) / 2) #! iterations for AnC

    iter = 0

    interlist = list((set(edges_of_g2)).intersection(set(edges_of_g1)))  # AnB
    random.shuffle(interlist)
    difflist = list(set(edges_of_g2) - set(edges_of_g1))  # B-A
    random.shuffle(difflist)
    print("Phase 1 generateG3")
    
    for AB in interlist:  # Select edges from (AnB) to map into U-(AuB)
        remain_edges = [i for i in interlist if i != AB]
        for CD in remain_edges:
            if g3.has_edge(*AB) and g3.has_edge(*CD) \
                    and (not (g1.has_edge(AB[0], CD[0]) or g2.has_edge(AB[0], CD[0]) or g3.has_edge(AB[0], CD[0]))) \
                    and (not (g1.has_edge(AB[1], CD[1]) or g2.has_edge(AB[1], CD[1]) or g3.has_edge(AB[1], CD[1]))):
                g3.remove_edges_from([AB, CD])
                g3.add_edges_from([(AB[0], CD[0]), (AB[1], CD[1])])
                iter += 1
                break

        if iter >= no_iterations: return g3
        elif iter >= no_iterations2: break #! break out of for loop if attained desired AnC

    print("Phase 2 generatedG3")
    for AB in difflist:  # Selecting edges from B-A to map into U-(AuB)
        remain_edges = [i for i in difflist if i != AB]
        for CD in remain_edges:
            if g3.has_edge(*AB) and g3.has_edge(*CD) \
                    and (not (g1.has_edge(AB[0], CD[0]) or g2.has_edge(AB[0], CD[0]) or g3.has_edge(AB[0], CD[0]))) \
                    and (not (g1.has_edge(AB[1], CD[1]) or g2.has_edge(AB[1], CD[1]) or g3.has_edge(AB[1], CD[1]))):
                g3.remove_edges_from([AB, CD])
                g3.add_edges_from([(AB[0], CD[0]), (AB[1], CD[1])])
                iter += 1
                break
        if iter >= no_iterations: return g3


def init_opi_assign(g, type='uniform'):  # assigns states
    for v in g:
        g.nodes[v]["state"] = random.uniform(0, 1)


def no_clusters(G1, d1, G2, d2, G3, d3):
    """ Remove links between nodes that never interact.
    """
    temG = copy.deepcopy(G1)

    for i in G1.edges():
        u, v = i[0], i[1]
        if abs(G1.nodes[u]["state"] - G1.nodes[v]["state"]) \
                + abs(G2.nodes[u]["state"] - G2.nodes[v]["state"]) * G2.has_edge(u, v) \
                + abs(G3.nodes[u]["state"] - G3.nodes[v]["state"]) * G3.has_edge(u, v) \
                > d1 + d2 * G2.has_edge(u, v) + d3 * G3.has_edge(u, v):  # ! two nodes never interact
            temG.remove_edge(u, v)

    return nx.number_connected_components(temG)


def plot_opi_distr(G):
    states = [G.nodes[i]["state"] for i in G]
    plt.hist(states, bins=list(np.linspace(0, 1, 50)))
    plt.show()
    y, x, _ = plt.hist(states, bins=list(np.linspace(0, 1, 50)))
    print("Maximum Freq. is " + str(y.max()))


def deffaunt3(graph1, d1, graph2, d2, graph3, d3, a, b):  # runs the deffaunt model, a,b are the nodes

    mu = 0.5
    x1 = graph1.nodes[a]["state"]
    y1 = graph1.nodes[b]["state"]
    if graph2.has_edge(a, b) == False and graph3.has_edge(a, b) == False:  # when nodes connected only on active layer
        if abs(x1 - y1) <= d1:
            xtemp = x1
            x1 = x1 + mu * (y1 - x1)
            y1 = y1 + mu * (xtemp - y1)

    elif graph2.has_edge(a, b) == True and graph3.has_edge(a, b) == False:  # connected on layer 2 but not 3
        a1 = graph2.nodes[a]["state"]
        b1 = graph2.nodes[b]["state"]
        if (abs(x1 - y1) + abs(a1 - b1)) <= (d1 + d2):
            xtemp = x1
            x1 = x1 + mu * (y1 - x1)
            y1 = y1 + mu * (xtemp - y1)

    elif graph2.has_edge(a, b) == False and graph3.has_edge(a, b) == True:  # connected on layer 3 but not 2
        a1 = graph3.nodes[a]["state"]
        b1 = graph3.nodes[b]["state"]
        if (abs(x1 - y1) + abs(a1 - b1)) <= (d1 + d3):
            xtemp = x1
            x1 = x1 + mu * (y1 - x1)
            y1 = y1 + mu * (xtemp - y1)

    else:  # connected on all layers
        a1 = graph2.nodes[a]["state"]
        b1 = graph2.nodes[b]["state"]
        c1 = graph3.nodes[a]["state"]
        d1 = graph3.nodes[b]["state"]
        if (abs(x1 - y1) + abs(a1 - b1) + abs(c1 - d1)) <= (d1 + d2 + d3):
            xtemp = x1
            x1 = x1 + mu * (y1 - x1)
            y1 = y1 + mu * (xtemp - y1)

    graph1.nodes[a]["state"] = round(x1, dec_places)  # reassign state values
    graph1.nodes[b]["state"] = round(y1, dec_places)


def run3(graph1, d1, graph2, d2, graph3, d3, e):  # runs deffaunt model for 3 layers, e is break parameter
    itera = 0
    opinion_dict1 = {}  # for time evolution create dict to track state of nodes per iteration
    opinion_dict2 = {}
    opinion_dict3 = {}
    toBreak = False  # condition for stability
    for i in list(graph1.nodes):
        opinion_dict1[i] = [graph1.nodes[i]["state"]]
        opinion_dict2[i] = [graph2.nodes[i]["state"]]
        opinion_dict3[i] = [graph3.nodes[i]["state"]]

    while True:
        itera += 1
        tempchoice = random.choice([1, 2, 3])  # picks active graph
        if tempchoice == 1:  # if graph1 is active graph
            nodex = random.choice(list(graph1.nodes))  # randomly pick 1 node
            xneigh = list(graph1.neighbors(nodex))  # find neighbours

        elif tempchoice == 2:  # if graph2 is active graph
            nodex = random.choice(list(graph2.nodes))  # randomly pick 1 node
            xneigh = list(graph2.neighbors(nodex))  # find neighbours

        else:  # if graph3 is active graph
            nodex = random.choice(list(graph3.nodes))  # randomly pick 1 node
            xneigh = list(graph3.neighbors(nodex))  # find neighbours

        if xneigh == []:  # error check if no neighbours
            continue
        nodey = random.choice(xneigh)  # pick a neighbour

        if tempchoice == 1:
            deffaunt3(graph1, d1, graph2, d2, graph3, d3, nodex, nodey)  # update using deffaunt3
            opinion_dict1[nodex] += [graph1.nodes[nodex]["state"]]
            opinion_dict1[nodey] += [graph1.nodes[nodey]["state"]]
        elif tempchoice == 2:
            deffaunt3(graph2, d2, graph1, d1, graph3, d3, nodex, nodey)  # update using deffaunt3
            opinion_dict2[nodex] += [graph2.nodes[nodex]["state"]]
            opinion_dict2[nodey] += [graph2.nodes[nodey]["state"]]
        else:
            deffaunt3(graph3, d3, graph2, d2, graph1, d1, nodex, nodey)  # update using deffaunt3
            opinion_dict3[nodex] += [graph3.nodes[nodex]["state"]]
            opinion_dict3[nodey] += [graph3.nodes[nodey]["state"]]

        if itera % (10 * nx.number_of_nodes(graph1)) == 0:  # check if stable at every 10*number of nodes iterations
            toBreak = True  # condition to break out of run
            for i in list(graph1.edges()):
                u, v = i[0], i[1]
                if abs(graph1.nodes[u]["state"] - graph1.nodes[v]["state"]) \
                        + abs(graph2.nodes[u]["state"] - graph2.nodes[v]["state"]) * graph2.has_edge(u, v) \
                        + abs(graph3.nodes[u]["state"] - graph3.nodes[v]["state"]) * graph3.has_edge(u, v) \
                        <= d1 + d2 * graph2.has_edge(u, v) + d3 * graph3.has_edge(u, v) \
                        and abs(graph1.nodes[u]["state"] - graph1.nodes[v]["state"]) >= e:
                    toBreak = False
                    break

            for j in list(graph2.edges()):
                u, v = j[0], j[1]
                if abs(graph2.nodes[u]["state"] - graph2.nodes[v]["state"]) \
                        + abs(graph1.nodes[u]["state"] - graph1.nodes[v]["state"]) * graph1.has_edge(u, v) \
                        + abs(graph3.nodes[u]["state"] - graph3.nodes[v]["state"]) * graph3.has_edge(u, v) \
                        <= d2 + d1 * graph1.has_edge(u, v) + d3 * graph3.has_edge(u, v) \
                        and abs(graph2.nodes[u]["state"] - graph2.nodes[v]["state"]) >= e:
                    toBreak = False
                    break

            for k in list(graph3.edges()):
                u, v = k[0], k[1]
                if abs(graph3.nodes[u]["state"] - graph3.nodes[v]["state"]) \
                        + abs(graph1.nodes[u]["state"] - graph1.nodes[v]["state"]) * graph1.has_edge(u, v) \
                        + abs(graph2.nodes[u]["state"] - graph2.nodes[v]["state"]) * graph2.has_edge(u, v) \
                        <= d3 + d1 * graph1.has_edge(u, v) + d2 * graph2.has_edge(u, v) \
                        and abs(graph3.nodes[u]["state"] - graph3.nodes[v]["state"]) >= e:
                    toBreak = False
                    break
        if toBreak == True:
            # print("Iterations for equilibrium: " + str(itera))
            break

    return [graph1, opinion_dict1, opinion_dict2, opinion_dict3]


if __name__ == "__main__":
    # ! net parameters:
    net_size = 5000
    k_min, k_max = 5, 80
    degree_distribution_type = "power law"
    degree_distr_para = 2  # power-law exponent

    print('Constructing G1...')
    graph1 = net_generator.network_construct(net_size, k_min, k_max, degree_distribution_type, degree_distr_para)
    graphs_dict = {}
    for j in [0.5,0.6,0.7,0.8,0.9]:
        alpha = beta = j
        print("Constructing G2...")
        temp = generate_G2(graph1, alpha)
        graph2 = temp[0]
        
        graphs_dict[j] = {}
        graphs_dict[j][1] = copy.deepcopy(graph1)
        graphs_dict[j][2] = copy.deepcopy(graph2)
        graphs_dict[j][3] = {}
        
        for i in np.arange(alpha+beta-1, alpha+0.1, 0.1):
            graph3 = generate_G3(graph2, graph1, beta, i, j)
            graphs_dict[j][3][round(i,1)] = copy.deepcopy(graph3)
            print("Testing: Alpha =" + str(round(eta_test(graph1, graph2),3)) + " Beta =" + str(round(eta_test(graph2, graph3),3)) + " Gamma =" + str(round(eta_test(graph1, graph3),3)))
            
    pickle_out = open("graphs2.pickle","wb")
    pickle.dump(graphs_dict,pickle_out)
    pickle_out.close()





