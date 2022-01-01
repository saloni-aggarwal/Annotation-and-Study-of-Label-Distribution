"""
This code combines the total label values for 25 annotated excel sheets of different students
"""
import os
from optparse import OptionParser
import pandas as pd
from scipy.stats import multinomial
import networkx as nx
import matplotlib.pyplot as plt
import htest


def plotHistograms(graphG):
    """
    This function finds the connected components and plot histogram for the size of components and degree of the graph
    :param graphG:
    """
    # Finding the connected components of the graph
    components = nx.connected_components(graphG)
    # print(components[0])

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


def makeGraph(labelsDic):
    """
    This function makes a graph for the given set of labels according to its message id
    :param labelsDic: Given dictionary of labels as value and message id and key
    :return: graph made
    """
    optparser = OptionParser()
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

    t = {}
    for x, y in labelsDic.items():
        if sum(y) == 0:
            break
        y1 = multinomial(options.sample_size, [yi / sum(y) for yi in y])
        y2 = htest.most_likely(y1)
        t[tuple(y2)] = x

    friendlist = []

    for x, y in labelsDic.items():
        if sum(y) == 0:
            break
        my = multinomial(options.sample_size, [yi / sum(y) for yi in y])
        mcr = htest.min_conf_reg(my, options.confidence)
        ldls = [tuple(i) for i in mcr]

        friends = set(ldls) & set(t.keys())
        for friend in friends:
            if tuple(y) != friend:
                friendlist.append((tuple(y), friend))
    g = nx.Graph(friendlist)
    graph2 = f"class_annotate_label_space_{options.confidence}_{options.sample_size}.gexf"
    nx.write_gexf(g, graph2)

    return g


def makingDict(dire):
    """
    This function makes two dictionaries: one containing the message id and the total labels for that message id and
    other dictionary contains the message id along with the message in the id
    :param dire: Directory containing all the annotated excel sheets
    :return: the two dictionaries made
    """
    # Declaring the dictionaries
    labelsDic = {}
    msgDic = {}
    # Reading the files in the directory that end with 'xlsx'
    for file in os.listdir(dire):
        if file.endswith(".xlsx"):
            # Reading the excel sheet that has title 'Data to Annotate'
            excelFile = pd.ExcelFile(open(dire + file, 'rb'))
            annotatedData = pd.read_excel(excelFile, 'Data to Annotate')
            # Replacing the NaN values with 0 in each column
            for column in annotatedData:
                annotatedData[column] = annotatedData[column].fillna(0)
            # Iterating over all the rows of the excel sheet
            for ind, data in annotatedData.iterrows():
                # Getting the message/item id for each message
                if 'Item Id' in data:
                    msgId = data['Item Id']
                else:
                    msgId = data['Message Id']
                # Getting the labels for each message and converting it into a list
                labels = data[2:].values.tolist()

                # Getting the message column
                M = data['Message']

                # If the id is already in the dictionary then adding the values of the labels and also if any person
                # mistyped the value except for 0 and 1 then changing the value as 0
                if msgId in labelsDic:
                    for i in range(0, len(labelsDic[msgId])):
                        if (isinstance(labels[i], str)):
                            labels[i] = 0
                        labelsDic[msgId][i] = labelsDic[msgId][i] + labels[i]

                # If the key is not in dictionary then if any person mistyped the value except for 0 and 1 then changing
                # the value as 0 and adding the labels and message in the respective dictionaries according to the id
                else:
                    for i in range(0, len(labels)):
                        if (isinstance(labels[i], str)):
                            labels[i] = 0
                    labelsDic[msgId] = labels
                    msgDic[msgId] = M

    # Returning both the dictionaries
    return labelsDic, msgDic


def combiningLabels():
    """
    Making the dataframe of the collected labels and message and then saving it as an excel sheet
    """
    # List of all the column names required in the final excel sheet
    columnNames = ["Item Id", "Message", "coming home from work", "complaining about work", "getting cut in hours",
                   "getting fired", "getting hired/job seeking", "getting promoted / raised", "going to work",
                   "losing job some other way", "none of the above but jobrelated", "not job related",
                   "offering support", "quitting a job"]
    # Making the labels and message dictionaries
    labelsDic, msgDic = makingDict("./AllFiles/")
    # Data file that will contain item id along with message and its label value
    dataFile = []

    # For each id in labelsDic
    for aId in labelsDic:
        # Making a tuple while appending id and message of that id
        data = {columnNames[0]: aId, columnNames[1]: msgDic[aId]}
        # Appending the label value in the data
        for i in range(2, len(columnNames)):
            data[columnNames[i]] = labelsDic[aId][i - 2]
        # Appending the data in data file list
        dataFile.append(data)
    # Making a dataframe of the list
    df = pd.DataFrame(dataFile, columns=columnNames)
    # Saving the dataframe as an excel sheet
    df.to_excel('jobQ3_both_dataset.xlsx', index=False)
    return labelsDic, msgDic


def graph_color(graphG, clusters):
    data = pd.read_csv(clusters)
    cluster = data['Clusters']
    # nx.set_node_attributes(graphG,"color")
    j = 0
    for i in graphG.nodes():
        graphG.nodes[i]["color"] = cluster[j]
        j += 1
    graph2 = f"graph2.gexf"
    nx.write_gexf(graphG, graph2)

    return graphG


def main():
    """
    Main function
    """
    labelsDic, msgDic = combiningLabels()
    g = makeGraph(labelsDic)
    # plotHistograms(g)
    # statistics(g)
    # cluster_graph = graph_color(g, 'initial_clusters.csv')
    cluster_graph2 = graph_color(g,'additional_label_clusters.csv')


if __name__ == '__main__':
    main()
