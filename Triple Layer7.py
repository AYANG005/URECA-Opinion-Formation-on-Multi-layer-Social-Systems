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
    pickle_in = open("graphs2.pickle","rb")
    graphs_dict = pickle.load(pickle_in)
    
    d2 = d3 = 0.2
    
    for i in [0.7]:#[0.5,0.6,0.7,0.8]:
        g1 = copy.deepcopy(graphs_dict[i][1])
        g2 = copy.deepcopy(graphs_dict[i][2])
        itera = -1
        
        for j in np.arange(2*i-1, i+0.1, 0.1):
            itera += 1
            g3 = copy.deepcopy(graphs_dict[i][3][round(j,1)])
            xaxis = [] #d1
            yaxis = [] #clusters
            
            for d1 in np.arange(0, 1.1, 0.1): #builds yaxis
                print("Running for alpha = beta = " + str(i) + " , gamma = " + str(round(j,1)) + " , d1 = " + str(round(d1,1)) + " , d2 = d3 = " + str(d2))
                g11 = copy.deepcopy(g1)
                g22 = copy.deepcopy(g2)
                g33 = copy.deepcopy(g3)
                init_opi_assign(g11) #assigns states
                init_opi_assign(g22)
                init_opi_assign(g33) 
            
                run3(g11, d1, g22, d2, g33, d3, state_epsilon) #run till equilibrium
                yaxis += [no_clusters(g33, d3, g22, d2, g11, d1)]
                xaxis += [round(d1,1)]
            col = ["b","g","r","c","k","m"][itera]
            plt.plot(xaxis, yaxis, label = "gamma= " + str(round(j,1)), fillstyle='none', marker = "s", color = col)
              
        plt.xlabel("d1")
        plt.ylabel("N Clusters")
        plt.legend(loc="best")  
        plt.title("Alpha=Beta= " + str(i))
        plt.savefig("alpha=beta=" + str(i) + " ,d2=d3= " +str(d2) + '.png')
        plt.close()
                
                





