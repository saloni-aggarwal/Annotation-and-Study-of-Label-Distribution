from collections import Counter
import networkx as nx
from optparse import OptionParser
import pandas as pd
from scipy.stats import multinomial
import matplotlib.pyplot as plt

import htest


def plotHistograms(graphG):
    """
    This function finds the connected components and plot histogram for the size of components and degree of the graph
    :param graphG:
    :return:
    """
    # Finding the connected components of the graph
    components = nx.connected_components(graphG)

    # Plotting histogram for the size of the generated connected components
    plt.hist([len(i) for i in components])
    plt.xlabel('Length')
    plt.ylabel('Number of Cluster')
    plt.title('Histogram for length of connected components')
    plt.show()

    # Plotting histogram for the degree of the nodes of given graph
    plt.hist(nx.degree_histogram(graphG))
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Histogram for degree of connected components')
    plt.show()


def statistics(graphG):
    """
    This function calculates the additional statistics of the graph
    :param graphG: The graph on which additional statistics is needed to be performed
    """
    density = nx.density(graphG)
    print('Density of the graph is ', density)
    triadic = nx.transitivity(graphG)
    print('triadic closure of the graph is ', triadic)
    degrees_node = dict(graphG.degree(graphG.nodes))
    degrees_node = sorted(degrees_node.items(), key=lambda x: x[1], reverse=True)
    for i in degrees_node[:10]:
        print(i)


def graph_color(graphG, clusters):
    data = pd.read_csv(clusters)
    cluster = data['Clusters']
    j = 0
    for i in graphG.nodes():
        graphG.nodes[i]["color"] = cluster[j]
        j += 1
    graph1 = f"graph1.gexf"
    nx.write_gexf(graphG, graph1)

    return graphG


optparser = OptionParser()
optparser.add_option('-f', '--inputFile',
                     dest='input_file',
                     help='json input filename',
                     default="jobQ3_BOTH_train.json")
optparser.add_option('-s', '--sample',
                     dest='sample_size',
                     help='Estimated sample size of each input',
                     default=10,
                     type='float')
optparser.add_option('-c', '--confidence',
                     dest='confidence',
                     help='Confidence (float) of regions desired',
                     default=0.9,
                     type='float')

(options, args) = optparser.parse_args()

df = pd.read_json(options.input_file, orient='split')
df.to_csv('aDataFrame.csv')
print('df = ', df)
Y_dict = (df.groupby('message_id')
          .apply(lambda x: dict(zip(x['worker_id'], x['label_vector'])))
          .to_dict())
print('Y_dict = ', Y_dict)
Ys = {x: list(y.values()) for x, y in Y_dict.items()}
Yz = {x: Counter(y) for x, y in Ys.items()}
dims = max([max(y.values()) for x, y in Yz.items()]) + 1
Y = {x: [Yz[x][i] if i in Yz[x] else 0 for i in range(dims)] for x, y in Yz.items()}
labels = df.groupby(['label', 'label_vector']).first().index.tolist()
Yframe = pd.DataFrame.from_dict(Y, orient='index')
XnY = df.groupby("message_id").first().join(Yframe, on="message_id")[['message', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

t = {}
for x, y in Y.items():
    y1 = multinomial(options.sample_size, [yi / sum(y) for yi in y])
    y2 = htest.most_likely(y1)
    t[tuple(y2)] = x

friendlist = []

for x, y in Y.items():
    print(f"x: {x}, y: {y}")
    my = multinomial(options.sample_size, [yi / sum(y) for yi in y])
    mcr = htest.min_conf_reg(my, options.confidence)
    ldls = [tuple(i) for i in mcr]
    friends = []
    """
    for mc in mcr:
        if tuple(mc) in t:
            friends.append(mc)
    """
    friends = set(ldls) & set(t.keys())
    for friend in friends:
        if (tuple(y) != friend):
            friendlist.append((tuple(y), friend))

g = nx.Graph(friendlist)
nx.write_gexf(g, f"label_space_{options.confidence}_{options.sample_size}.gexf")
g = nx.read_gexf('label_space_0.9_10.gexf')
mylist = []

for n in nx.nodes(g):
    n = n[1:len(n)-1]
    # print(n)
    # n = list(n)
    points = n.split(",")
    mylist.append(list(map(float, points)))
clusters = pd.DataFrame(mylist)
clusters.to_csv('initial_clusters.csv')
plotHistograms(g)
statistics(g)
cluster_graph = graph_color(g, 'initial_cluster1.csv')
