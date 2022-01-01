from collections import Counter
import networkx as nx
from optparse import OptionParser
import pandas as pd
from scipy.stats import multinomial

import htest

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
    # my = multinomial(sum(y), [yi/sum(y) for yi in y])
    my = multinomial(options.sample_size, [yi / sum(y) for yi in y])
    mcr = htest.min_conf_reg(my, options.confidence)
    # ldls = [[int(i) for i in m.p * m.n] for m in mcr]
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
