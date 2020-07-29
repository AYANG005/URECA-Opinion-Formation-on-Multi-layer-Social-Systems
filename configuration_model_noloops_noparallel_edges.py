import heapq
from itertools import combinations, permutations
import math
from operator import itemgetter
import random
import networkx as nx
from networkx.utils import random_weighted_sample

def configuration_model(deg_sequence, create_using=None, seed=None):
    """ Return a random graph with the given degree sequence.

    The configuration model generates a random pseudograph (graph with
    parallel edges and self loops) by randomly assigning edges to
    match the given degree sequence.

    Parameters
    ----------
    deg_sequence :  list of integers
        Each list entry corresponds to the degree of a node.
    create_using : graph, optional (default MultiGraph)
       Return graph of this type. The instance will be cleared.
    seed : hashable object, optional
        Seed for random number generator.

    Returns
    -------
    G : MultiGraph
        A graph with the specified degree sequence.
        Nodes are labeled starting at 0 with an index
        corresponding to the position in deg_sequence.

    Raises
    ------
    NetworkXError
        If the degree sequence does not have an even sum.

    See Also
    --------
    is_valid_degree_sequence

    Notes
    -----
    As described by Newman [1]_.

    A non-graphical degree sequence (not realizable by some simple
    graph) is allowed since this function returns graphs with self
    loops and parallel edges.  An exception is raised if the degree
    sequence does not have an even sum.

    This configuration model construction process can lead to
    duplicate edges and loops.  You can remove the self-loops and
    parallel edges (see below) which will likely result in a graph
    that doesn't have the exact degree sequence specified.

    The density of self-loops and parallel edges tends to decrease
    as the number of nodes increases. However, typically the number
    of self-loops will approach a Poisson distribution with a nonzero
    mean, and similarly for the number of parallel edges.   Consider a
    node with k stubs. The probability of being joined to another stub of
    the same node is basically (k-1)/N where k is the degree and N is
    the number of nodes. So the probability of a self-loop  scales like c/N
    for some constant c.  As N grows, this means we expect c self-loops.
    Similarly for parallel edges.

    References
    ----------
    .. [1] M.E.J. Newman, "The structure and function of complex networks",
       SIAM REVIEW 45-2, pp 167-256, 2003.

    .. NetworkX

    Examples
    --------

    >>> from networkx.utils import powerlaw_sequence
    >>> z=nx.utils.create_degree_sequence(100,powerlaw_sequence)
    >>> G=nx.configuration_model(z)

    
    To remove parallel edges:

    >>> G=nx.Graph(G)

    To remove self loops:

    >>> G.remove_edges_from(G.selfloop_edges())
    """
    if not sum(deg_sequence) % 2 == 0:
        raise nx.NetworkXError('Invalid degree sequence')

    if create_using is None:
        create_using = nx.MultiGraph()
    elif create_using.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    if not seed is None:
        random.seed(seed)

    # start with empty N-node graph
    N = len(deg_sequence)

    # allow multiedges and selfloops
    G = nx.empty_graph(N, create_using)

    if N == 0 or max(deg_sequence) == 0:  # done if no edges
        return G

        # build stublist, a list of available degree-repeated stubs
    # e.g. for deg_sequence=[3,2,1,1,1]
    # initially, stublist=[1,1,1,2,2,3,4,5]
    # i.e., node 1 has degree=3 and is repeated 3 times, etc.
    stublist = []
    for n in G:
        for i in range(int(deg_sequence[n])):
            stublist.append(n)

    # shuffle stublist and assign pairs by removing 2 elements at a time
    random.shuffle(stublist)
    while stublist:

        # n1, n2 = random.sample(stublist,2)
        n1 = stublist[0]
        n2 = random.sample(stublist,1)[0]

        if n1 != n2 and not G.has_edge(n1, n2):
            G.add_edge(n1, n2)
            stublist.remove(n1)
            stublist.remove(n2)

        """
        n1 = stublist.pop()
        n2 = stublist.pop()
        G.add_edge(n1, n2)
        """

    G.name = "configuration_model %d nodes %d edges" % (G.order(), G.size())
    return G