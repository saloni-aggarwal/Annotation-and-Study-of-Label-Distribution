1. The file jobQ3a_lm7819.csv represents the annotated data by lm7819 and jobQ3a_sna2485.xlsx represents the annotated data by sna2485.
2. The file jobQ3_both_dataset.csv represents the annotated data with additional labels collected from 25 spreadsheets.
3. The file initial_clusters.csv represents the initial annotated data along with cluster of each item. 
4. The file additional_label_clusters.csv the cluster each item for the data with additional labels.
5. When executed the file project2.py it makes histogram for graph1.gexf file and also does some additional statistical analysis for graph1. In addition to this it also adds color to the clusters formed.
6. When executed visualization.py file it calculates cohen's kappa score and makes covariance matrix for each column in graph1.gexf file.
7. When executed, the file clustering.py performs clustering on the annotated data and prints the top 10 frequent words per cluster.
8. When executed labelAddition.py file it adds additional labels and makes graph for this dataset. Then considering this graph it computes additional statistical computations along with making histograms and also it calculates cohen's kappa score and covariance matrix for each column of this graph called as graph2.gexf.
9. Also the file additional_label_clusters contain the cluster for graph2 and initial_clusters.csv contains cluster for JobQ3_Both_train.json data.
10. label_space_0.9_10.gexf is a graph1 file and class_annotate_label_space_0.9_10.gexf is graph2 file.
