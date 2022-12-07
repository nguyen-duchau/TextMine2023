import pickle, json 

import nltk, spacy
import numpy as np
import pandas as pd  
import operator as op 
import itertools as it, functools as ft 
 
import networkx as nx

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data

from transformers import AutoModel, AutoTokenizer
from modelization import PositionalEncoding

from os import path 
from glob import glob
from unidecode import unidecode
from time import time 
from math import ceil 
import re
import html

from nltk.cluster import KMeansClusterer
from sentence_transformers import SentenceTransformer
from libraries.log import logger
from rich.progress import track

graph_neural_network_config = {
    'layers_config': [512, 512, 512, 512], 
    'activations': [1, 1, 0], 
    'drop_val': 0.1
}

map_serializer2mode = {
    json: ('r', 'w'), 
    pickle: ('rb', 'wb')
}

def measure(func):
    @ft.wraps(func)
    def _measure(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            duration = end_ if end_ > 0 else 0
            logger.debug(f"{func.__name__:<20} total execution time: {duration:04d} ms")
    return _measure

def serialize(path2location, data, serializer=pickle):
    mode = map_serializer2mode.get(serializer, None)
    if mode is None:
        raise ValueError(f'serializer option must be pickle or json')
    with open(path2location, mode=mode[1]) as fp:
        serializer.dump(data, fp)

def deserialize(path2location, serializer=pickle):
    mode = map_serializer2mode.get(serializer, None)
    if mode is None:
        raise ValueError(f'serializer option must be pickle or json')
    with open(path2location, mode=mode[0]) as fp:
        data = serializer.load(fp)
        return data 

def build_response(response_status, error_message, data={}):
    return json.dumps({
        'global_status': response_status, 
        'error_message': error_message, 
        'response': data
    }).encode()

def to_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [ sent.strip().lower() for sent in sentences ]
    return sentences

def to_chunks(sentences, chunk_size=7):
    chunks = []
    nb_sentences = len(sentences)
    for idx in range(0, nb_sentences, chunk_size):
        chunks.append('\n'.join(sentences[idx:idx+chunk_size]))
    return chunks 

def textrank(matrix, damp=0.85, nb_iters=256, tol=0.001):
    nb_nodes = matrix.shape[0] 
    node_ids = np.arange(nb_nodes)
    node_ranks = np.full((nb_nodes, 1), 1 / nb_nodes) 
    normalized_matrix = matrix / np.sum(matrix, axis=1)[:, None]
    
    cursor = 0
    keep_ranking = True 
    while keep_ranking and cursor < nb_iters:
        predicted_node_ranks = (1 - damp) + damp * normalized_matrix.T @ node_ranks
        distance = np.sqrt( np.sum((predicted_node_ranks - node_ranks) ** 2) + 1e-8 )
        keep_ranking = distance > tol
        node_ranks = predicted_node_ranks
        cursor += 1
    # end while loop ...! 
    
    return list(zip(node_ids, node_ranks)) 

def ranking(sentence_embeddings, weight=0.3):
    nb_sentences = sentence_embeddings.shape[0]
    summary_size = ceil(weight * nb_sentences)

    scores = sentence_embeddings @ sentence_embeddings.T 
    sentence_norms = np.linalg.norm(sentence_embeddings, axis=1)
    expanded_sentence_norms = sentence_norms[:, None] * sentence_norms[None, :]
    weighted_scores = scores / expanded_sentence_norms

    weighted_scores += np.random.normal(0.0, 0.4, size=weighted_scores.shape)

    max_weighted_scores = np.max(weighted_scores, axis=1)[:, None]
    min_weighted_scores = np.min(weighted_scores, axis=1)[:, None]
    
    normalized_scores = (weighted_scores - min_weighted_scores) / (max_weighted_scores - min_weighted_scores)
    normalized_scores[normalized_scores < 0.2] = 0.0

    node_ranks = textrank(normalized_scores, 0.85, 128, 0.001)
    sorted_node_by_ranks = sorted(node_ranks, key=op.itemgetter(1), reverse=True)  # descending rank order 

    selected_nodes_ranks = sorted_node_by_ranks[:summary_size]  # took top_k ranks 
    selected_nodes = list(map(op.itemgetter(0), selected_nodes_ranks))
    sorted_selected_nodes = sorted(selected_nodes)

    return sorted_selected_nodes

def vectorize(data, vectorizer, device='cpu', to_tensor=True):
    with th.no_grad():
        output = vectorizer(**data)['last_hidden_state'].squeeze(0)
    return output

def get_offsets_dict(offsets_mapping):
    output_dict={}
    counter=-1
    for i, (start, end) in enumerate(offsets_mapping):
        if start==0:
            counter+=1
            output_dict[counter]=[]
        output_dict[counter].append(i)
    return output_dict

def build_graph(embeddings_matrix, edges, distance, entity_pair, label, index, dependency_edges, spacy_tokens, offsets_mapping):
    
    nb_nodes = embeddings_matrix.shape[0]
    edges = []
    for node in range(nb_nodes):
        for dist in range(-distance, distance+1):
            edges.append((node, min(max(node+dist, 0),nb_nodes-1)))
    edges = list(set(edges))
    edge_index = th.tensor(list(zip(*edges)))

    return Data(embedding=embeddings_matrix, edge_index=edge_index, num_nodes=nb_nodes, label=label, idx=index)

def load_vectorizer(path2vectorizer, model_name=None):
    if path.isfile(path2vectorizer):
        tokenizer, vectorizer = deserialize(path2vectorizer)
        logger.success('the vectorizer was loaded')
    else:
        try:
            if not model_name:
                _, model_name = path.split(path2vectorizer)
            tokenizer, vectorizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True), AutoModel.from_pretrained(model_name)
            serialize(path2vectorizer, [tokenizer, vectorizer])
            logger.success(f'the vectorizer was downloaded and saved at {path2vectorizer}')
        except Exception as e:
            logger.error(e)
            raise ValueError(f'{path2vectorizer} is not a valid path')
    return tokenizer, vectorizer

def scoring(fingerprint, fingerprint_matrix):
    scores = fingerprint @ fingerprint_matrix.T 
    X = np.linalg.norm(fingerprint)
    Y = np.linalg.norm(fingerprint_matrix, axis=1)
    W = X * Y
    weighted_scores = scores / W 
    return weighted_scores

def load_dataset(path2bulk, save_path=None):
    if not save_path:
        save_path=path2bulk+'.pkl'
    if path.isfile(save_path):
        return deserialize(save_path, pickle)
    with open(path2bulk, 'r') as f:
        data=json.load(f)
    keys = ['identifier', 'text', 'annotations']
    datadict = {k:[] for k in keys}
    for i, _ in enumerate(data):
        for key in keys:
            datadict[key].append(data[i][key])
    df = pd.DataFrame.from_dict(datadict)
    df = df.sort_values(by=['identifier'])
    return df

def add_key_and_return(dic, key, value):
    dic[key] = value
    return dic

def bounds_overlap(interval1, interval2):
    (s1, e1), (s2, e2) = interval1, interval2
    b1, b2 = set(range(s1, e1)), set(range(s2, e2))
    return bool(b1.intersection(b2))

def tokenid_to_charid(idx, tokenslist):
    return len(' '.join(tokenslist[:idx]))+1

def build_graph_edges(sentence_analysis):
    edges = list(it.chain(*[[(token.i,child.i) for child in token.children] for token in sentence_analysis]))
    edgeslist = it.chain(*edges)
    root = [token.i for token in sentence_analysis if token.dep_=='ROOT']
    missing_edges = [(root[0], i) for i in range(len(sentence_analysis)) if i not in edgeslist]
    return set(list(edges) + missing_edges)

def get_entity_types(sentence_analysis, ids1, ids2, tokenslist):
    type1, type2 = 'UNK', 'UNK'
    bounds1 = (tokenid_to_charid(ids1[0], tokenslist), tokenid_to_charid(ids1[-1], tokenslist))
    bounds2 = (tokenid_to_charid(ids2[0], tokenslist), tokenid_to_charid(ids2[-1], tokenslist))
    for ent in sentence_analysis.ents:
        if bounds_overlap(bounds1, (ent.start_char, ent.end_char)):
            type1=ent.label_
        if bounds_overlap(bounds2, (ent.start_char, ent.end_char)):
            type2=ent.label_
    return type1, type2

def get_dependency_edges(analysis):
    edges = []
    for token in analysis:
        for child in token.children:
            edges.append((token.i,child.i))
    edgeslist = list(set(it.chain(*edges)))
    root = [i for i,tok in enumerate(analysis) if tok.dep_=='ROOT' and i in edgeslist][0]
    for e in range(len(analysis)):
        if e not in edgeslist:
            edges.append((root, e))
    return edges

def clean_str(string):
    string = re.sub('<.+?>', '', string)
    return html.unescape(string)
