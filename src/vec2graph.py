#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import heapq  # Heap - Priority Queue
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from math import cos, pi
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration /
                                                            float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def load_dataset(dataset_filename):
    return pd.read_csv(dataset_filename, sep=";",
                       header=None,
                       names=["id", "class", "text"],
                       index_col=0)


def build_bridges2(G: nx.Graph, wvs, sims_df) -> nx.Graph:
    connected_G = G.copy()
    connected_components = \
        [list(cc) for cc in nx.connected_components(connected_G)]
    num_ccs = len(connected_components)
    initial_num_css = num_ccs
    print("Number of initial connected components: " + str(initial_num_css))
    if initial_num_css == 1:
        return connected_G
    n1s = []
    n2s = []
    cc1s = []
    cc2s = []
    sims = []
    num_components_processed = 0
    printProgressBar(num_components_processed,
                     num_ccs,
                     "Calculating similarities",
                     "Complete")
    for i in range(num_ccs - 1):
        for j in range(i + 1, num_ccs):
            cc1 = connected_components[i]
            cc2 = connected_components[j]
            for n1 in cc1:
                for n2 in cc2:
                    n1s.append(n1)
                    n2s.append(n2)
                    cc1s.append(i)
                    cc2s.append(j)
                    try:
                        sim = sims_df[n1][n2]
                    except Exception:
                        sim = -1000.0
                    sims.append(sim)
        num_components_processed += 1
        printProgressBar(num_components_processed,
                         num_ccs,
                         "Calculating similarities",
                         "Complete")
    print("")
    C = pd.DataFrame({'n1': n1s,
                      'n2': n2s,
                      'cc1': cc1s,
                      'cc2': cc2s,
                      'sim': sims})
    edges2add = []
    max_merges = num_ccs - 1
    num_merges = 0
    printProgressBar(num_merges, max_merges, "Merging components", "Complete")
    C.sort_values(by=['sim'], ascending=False, inplace=True)
    while num_ccs > 1:
        C = C[C['cc1'] != C['cc2']]
        selected_edge = C.iloc[0]
        n1 = selected_edge.get(key='n1')
        n2 = selected_edge.get(key='n2')
        cc1 = selected_edge.get(key='cc1')
        cc2 = selected_edge.get(key='cc2')
        sim = selected_edge.get(key='sim')
        edges2add.append((n1, n2, sim))
        C = C.iloc[1:]
        C.replace({'cc2': {cc2: cc1}}, inplace=True)
        num_ccs -= 1
        num_merges += 1
        printProgressBar(num_merges,
                         max_merges,
                         "Merging components",
                         "Complete")
    print("Adjusting graph... ", end="", flush=True)
    for n1, n2, sim in edges2add:
        connected_G.add_edge(n1, n2,
                             weight=sim,
                             distance=pow(cos(sim), -1.0) / pi)
    print("OK!")
    return connected_G


def build_bridges(G: nx.Graph, wvs, sims_cache) -> nx.Graph:
    connected_G = G.copy()
    nx.connected
    connected_components = [list(cc)
                            for cc in nx.connected_components(connected_G)]
    num_ccs = len(connected_components)
    initial_num_css = num_ccs
    print("Number of initial connected components: " + str(initial_num_css))
    clusters_merged = 0
    if initial_num_css > 1:
        printProgressBar(clusters_merged,
                         initial_num_css,
                         "Making a connected graph",
                          "Complete")
    while num_ccs > 1:
        max_sim = -1000.0
        node1 = None
        node2 = None
        for i in range(num_ccs - 1):
            for j in range(i + 1, num_ccs):
                cc1 = connected_components[i]
                cc2 = connected_components[j]
                for n1 in cc1:
                    for n2 in cc2:
                        if (n1, n2) in sims_cache:
                            sim = sims_cache[(n1, n2)]
                        else:
                            try:
                                sim = wvs.similarity(n1, n2)
                            except:
                                sim = -1000.0
                        if sim > max_sim:
                            max_sim = sim
                            node1 = n1
                            node2 = n2
        if (node1 is None) or (node2 is None):
            print((node1, node2, max_sim))
            sys.exit(1)
        if (node1 is not None) and (node2 is not None):
            connected_G.add_edge(node1, node2, weight=max_sim, distance=pow(cos(max_sim), -1.0)/pi)
        connected_components = [list(cc) for cc in nx.connected_components(connected_G)]
        num_ccs = len(connected_components)
        clusters_merged += 1
        printProgressBar(clusters_merged, initial_num_css, "Making a connected graph", "Complete")
    printProgressBar(clusters_merged+1, initial_num_css, "Making a connected graph", "Complete")
    return connected_G


def get_word_graph2(corpus_vocab, wvs, threshold):
    vocab_size = len(corpus_vocab)
    wvs_target = np.array([wvs[w] for w in corpus_vocab])
    print("Calculating similarities... ", end="")
    similarities = cosine_similarity(wvs_target, dense_output=False)
    adj_mat = np.copy(similarities)
    np.fill_diagonal(adj_mat, 0)
    adj_mat[adj_mat < threshold] = 0
    print("OK!")
    print("Creating word graph... ", end="")
    adj_mat_df = pd.DataFrame(adj_mat)
    adj_mat_df.columns = corpus_vocab
    adj_mat_df.index = corpus_vocab
    sims_df = pd.DataFrame(similarities)
    sims_df.columns = corpus_vocab
    sims_df.index = corpus_vocab
    G = nx.from_pandas_adjacency(adj_mat_df)
    print("OK!")
    G = build_bridges2(G, wvs, sims_df)
    return G


def get_word_graph(corpus_vocab, wvs, threshold):
    word_graph = nx.Graph()
    word_graph.add_nodes_from(corpus_vocab)
    vocab_size = len(corpus_vocab)
    num_pot_edges = vocab_size * (vocab_size - 1) / 2
    edges_processed = 0
    sims_cache = dict()
    printProgressBar(edges_processed,
                     num_pot_edges,
                     prefix='Word graph',
                     suffix='Complete')
    for i in range(vocab_size - 1):
        for j in range(i + 1, vocab_size):
            try:
                sim = wvs.similarity(corpus_voc ab[i], corpus_vocab[j])
                sims_cache[(corpus_vocab[i], corpus_vocab[j])] = sim
                sims_cache[(corpus_vocab[j], corpus_vocab[i])] = sim
                if sim >= threshold:
                    word_graph.add_edge(corpus_vocab[i],
                                        corpus_vocab[j],
                                        weight=sim,
                                        distance=pow(cos(sim), -1.0) /
                                        pi)
            except Exception:
                pass
            edges_processed += 1
            printProgressBar(edges_processed,
                             num_pot_edges,
                             prefix='Word graph',
                             suffix='Complete')
    word_graph = build_bridges2(word_graph, wvs)
    return word_graph


def build_bridges3(corpus_vocab, word_graph, sims_cache):
    # Build map of node to connected component
    connected_components = \
        [list(map(lambda x: corpus_vocab.index(x), list(cc)))
         for cc in nx.connected_components(word_graph)]
    initial_component_number = len(connected_components)
    vocabulary_length = len(corpus_vocab)
    # Map from node to component ID
    node_component = np.zeros(vocabulary_length)
    for index, cc in enumerate(connected_components):
        node_component[cc] = index  # mark node as associated to CC

    # Sort the similarity cache tuples, in descending order
    # Edges with higher similarity go to the beginning
    sims_cache.sort(reverse=True)

    # Check components information
    unique_components = np.unique(node_component).shape[0]
    assert(initial_component_number == unique_components)
    edges_processed = 0
    joined_components = 0
    initial_edge_number = len(sims_cache)
    printProgressBar(joined_components,
                     initial_component_number,
                     prefix='Joining components',
                     suffix='Complete')
    # while there are edges in heap or more than 1 component
    while unique_components > 1:
        largest_weight_edge = sims_cache[edges_processed]
        sim = largest_weight_edge[0]
        node_i, node_j = largest_weight_edge[1]  # unpack nodes
        edges_processed += 1
        if node_component[node_i] != node_component[node_j]:
            # Add edge to graph
            word_graph.add_edge(corpus_vocab[node_i],
                                corpus_vocab[node_j],
                                weight=largest_weight_edge[0],
                                distance=(1.0 / (pi * cos(sim))))
            # Union find
            # Find other nodes in component of node j
            nodes_in_component_j = \
                np.where(node_component == node_component[node_j])[0]
            # Set these nodes to the same component of node i
            node_component[nodes_in_component_j] = node_component[node_i]
            # Recalculate unique components
            unique_components = np.unique(node_component).shape[0]
            joined_components += 1
            printProgressBar(joined_components,
                             initial_component_number,
                             prefix='Joining components',
                             suffix='Complete')
    return word_graph


def get_word_graph3(corpus_vocab, wvs, threshold):
    vocab_size = len(corpus_vocab)
    wvs_target = np.array([wvs[w] for w in corpus_vocab])
    print("Building graph... ", end="")
    similarities = cosine_similarity(wvs_target, dense_output=False)
    adj_mat = similarities
    # Zero out self-similarities
    np.fill_diagonal(adj_mat, 0)
    # Find similarities below threshold
    below_threshold = np.where(similarities < threshold)
    # Create cache of removed edges
    sims_cache = []
    for node_i, node_j in zip(below_threshold[0], below_threshold[1]):
        sims_cache.append((similarities[node_i, node_j],
                           (node_i, node_j)))
    # Zero out similarities below threshold
    adj_mat[adj_mat < threshold] = 0
    # Create DataFrame from np.ndarray
    adj_mat_df = pd.DataFrame(adj_mat)
    adj_mat_df.columns = corpus_vocab
    adj_mat_df.index = corpus_vocab
    # Create graph
    word_graph = nx.from_pandas_adjacency(adj_mat_df)
    print("OK!")
    # Connect graph
    word_graph = build_bridges3(corpus_vocab, word_graph, sims_cache)
    print()
    if nx.is_connected(word_graph):
        print('Word graph is connected!')
    return word_graph


def shortest_path(cc1, cc2, word_graph):
    word_graph_aux = word_graph.copy()
    word_graph_aux.add_node("cc1")
    word_graph_aux.add_node("cc2")
    for n in cc1:
        word_graph_aux.add_edge("cc1", n, weight=0.0, distance=1.0)
    for n in cc2:
        word_graph_aux.add_edge("cc2", n, weight=0.0, distance=1.0)
    shortest_path_nodes = nx.shortest_path(word_graph_aux,
                                           source="cc1", target="cc2",
                                           weight="distance")
    shortest_path_nodes.remove("cc1")
    shortest_path_nodes.remove("cc2")
    for n in cc1:
        try:
            shortest_path_nodes.remove(n)
        except Exception:
            continue
    for n in cc2:
        try:
            shortest_path_nodes.remove(n)
        except Exception:
            continue

    return (shortest_path_nodes,
            word_graph.subgraph(shortest_path_nodes).size(weight='weight'))


def export_doc_graph(words, word_graph, doc_graph_fn):
    doc_graph = word_graph.subgraph(words)
    nx.write_gexf(doc_graph, doc_graph_fn + "_original")
    connected_components = \
        [list(cc) for cc in nx.connected_components(doc_graph)]
    connected_components.sort(key=len)
    num_connected_components = len(connected_components)

    while num_connected_components > 1:
        shortest_path_nodes = None
        shortest_path_weight = -1000.0
        for i in range(1, num_connected_components):
            spn, spw = shortest_path(connected_components[0],
                                     connected_components[i],
                                     word_graph)
            if spw > shortest_path_weight:
                shortest_path_nodes = spn
                shortest_path_weight = spw
        doc_graph = \
            word_graph.subgraph(list(doc_graph.nodes()) +
                                shortest_path_nodes)
        connected_components = \
            [list(cc) for cc in nx.connected_components(doc_graph)]
        num_connected_components = len(connected_components)

    nx.write_gexf(doc_graph, doc_graph_fn)
    return list(doc_graph.nodes())


def main(argv=None):

    # Command line arguments
    if argv is None:
        argv = sys.argv

    ds_fn = argv[1]
    wv_fn = argv[2]
    threshold = float(argv[3])
    wg_fn_prefix = argv[4]

    # Dataset
    print("Loading corpus...", end=" ", flush=True)
    corpus = None
    ds = load_dataset(ds_fn)
    corpus = ds['text'].tolist()
    Y = ds['class'].tolist()
    print("OK!")

    # Vocabulary
    print("Extracting vocabulary...", end=" ", flush=True)
    corpus_vocab = set()
    for doc in corpus:
        for word in doc.split():
            corpus_vocab.add(word)
    corpus_vocab = list(corpus_vocab)
    corpus_len = len(corpus)
    print("OK! (initial vocab size: " + str(len(corpus_vocab)) + ")")

    # Loading word vectors
    print("Loading word vectors...", end=" ", flush=True)
    wvs = KeyedVectors.load_word2vec_format(wv_fn, binary=False)
    print("OK!")

    # Remove words that haven't vectors
    corpus_vocab = list(set(corpus_vocab) & set(wvs.vocab.keys()))
    corpus_vocab.sort()
    print("Final vocabulary size: " + str(len(corpus_vocab)))

    # Word graph
    wg_gexf_fn = wg_fn_prefix + ".gexf"
    wg_pkl_fn = wg_fn_prefix + ".pkl"
    word_graph = None
    if (not os.path.exists(wg_gexf_fn)) or True:
        word_graph = get_word_graph3(corpus_vocab, wvs, threshold)
        # Saving word graph
        print("Saving word graph...", end=" ", flush=True)
        nx.write_gexf(word_graph, wg_gexf_fn)
        with open(wg_pkl_fn, "wb") as wg_pkl_file:
            pickle.dump(word_graph, wg_pkl_file)
        print("OK!")
    else:
        print("Loading word graph...", end=" ", flush=True)
        with open(wg_pkl_fn, "rb") as wg_pkl_file:
            word_graph = pickle.load(wg_pkl_file)
        print("OK!")


if __name__ == '__main__':
    main()
