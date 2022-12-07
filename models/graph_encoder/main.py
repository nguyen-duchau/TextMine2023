import click
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from os import getenv 

from rich.progress import track 
from libraries.log import logger 
from libraries.strategies import *
from data_holder import VDSDataset 
from modelization import SimpleGCN

from torch_geometric.loader import DataLoader

@click.group(chain=False, invoke_without_command=True)
@click.pass_context
def router(ctx):
    ctx.ensure_object(dict)
    logger.debug('nltk initialization ...!')

    path2models = getenv('MODELS')
    path2dataset = getenv('DATASET')
    path2features = getenv('FEATURES')
    path2nltk_data = getenv('NLTK_DATA')
    path2checkpoints = getenv('CHECKPOINTS')
    
    logger.debug('env variables loading')

    ctx.obj['path2models'] = path2models
    ctx.obj['path2dataset'] = path2dataset 
    ctx.obj['path2features'] = path2features  
    ctx.obj['path2nltk_data'] = path2nltk_data 
    ctx.obj['path2checkpoints'] = path2checkpoints

    logger.debug('nltk initialization')
    nltk.download('punkt', download_dir=path2nltk_data) 
    nltk.download('stopwords', download_dir=path2nltk_data)
    
    ctx.ensure_object(dict)
    subcommand = ctx.invoked_subcommand
    if subcommand is None:
        logger.debug('use --help option to see availables options')
    else:
        logger.debug(f'{subcommand} was called')

@router.command()
@click.option('--vectorizer', help='vectorizer name(feature extraction)', type=str)
@click.option('--corpus_filename', help='corpus file name', type=str)
@click.option('--graph_filename', help='features serialization file name', type=str)
@click.pass_context
def preprocessing(ctx, vectorizer, corpus_filename, graph_filename):
    try:
        device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        path2models = ctx.obj['path2models']
        path2dataset = ctx.obj['path2dataset']
        path2features = ctx.obj['path2features']

        path2corpus = path.join(path2dataset, corpus_filename)
        if '/' in vectorizer:
            path2vectorizer = path.join(path2models, ''.join(vectorizer.split('/')))
        else:
            path2vectorizer = path.join(path2models, vectorizer)
        path2_graphs = path.join(path2features, graph_filename)

        tokenizer, vectorizer = load_vectorizer(path2vectorizer, vectorizer)
        embedding_model = vectorizer.embeddings
        corpus = load_dataset(path2corpus)

        global_graph_accumulator = []
        documents = corpus['text'].tolist()
        labels = corpus['annotations'].tolist()
        ids = corpus['identifier'].tolist()
        # For each document in corpus
        for doc_id, doc in enumerate(track(documents, f'processing graphs...')):
            tokens = tokenizer(doc, return_tensors='pt', return_offsets_mapping=True, is_split_into_words=True)
            tokens, offsets_mapping = {k:v for k,v in tokens.items() if k != 'offset_mapping'}, tokens['offset_mapping']

            embeddings_matrix = embedding_model(tokens['inout_ids']).cpu()

            graph = build_graph(embeddings_matrix, 2, labels[doc_id], ids[doc_id], offsets_mapping)

            global_graph_accumulator.append(graph)

        serialize(path2_graphs, global_graph_accumulator, pickle)
        logger.success(f'{len(global_graph_accumulator):03d} dgl graphs were saved...!')
    except Exception as e:
        logger.error(e)
    
@router.command()
@click.option('--graph_filename', help='', type=str)
@click.option('--nb_epochs', type=int, default=32)
@click.option('--batch_size', type=int, default=16)
@click.option('--checkpoint', type=str)
@click.option('--period', default=8, type=int)
@click.option('--print_period', default=32, type=int)
@click.pass_context
def train(ctx, graph_filename, nb_epochs, batch_size, checkpoint, period, print_period):
    
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    logger.debug(f'target device : {device}')

    path2features = ctx.obj['path2features']
    path2checkpoints = ctx.obj['path2checkpoints']

    path2graph_filename = path.join(path2features, graph_filename)
    
    train_dataset = VDSDataset(path2graph_filename, begin_percentage=0.0, end_percentage=0.75)
    val_dataset = VDSDataset(path2graph_filename, begin_percentage=0.75, end_percentage=100.0)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    logger.debug('dataset and dataloader were initialized')

    path2initial_weights = path.join(path2checkpoints, checkpoint)
    if path.isfile(path2initial_weights):
        logger.debug(f'initial weights were detected, the model will use {path2initial_weights}')
        model = th.load(path2initial_weights, map_location=device)
    else:
        logger.debug('no initial weights were detected, the model will be created from scratch')
        model = SimpleGCN(train_dataset)
        model.to(device)

    print(model)
    logger.debug('model was initialized')

    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    logger.debug('training will start in 1 s')
    
    nb_graphs = len(train_dataset)
    mean_val_loss = 100.0
    val_f1=0.0
    for epoch in range(nb_epochs):
        cursor = 0
        for graph_batch in train_dataloader:
            crr_bt_size = graph_batch.label.shape[0]
            cursor += crr_bt_size
            model.train()
            optimizer.zero_grad()
            pred = model(graph_batch)
            loss = criterion(pred, graph_batch.label)
            loss.backward()
            optimizer.step()
            if int(cursor/batch_size)%print_period == 0:
                logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{cursor:05d}/{nb_graphs:05d}] >> Loss: train {loss:07.3f} | val {mean_val_loss:07.3f} >> F1: {val_f1:07.3f}')
        val_loss_accumulator = []
        val_pred_accumulator, val_label_accumulator = [], []
        for graph_batch in val_dataloader:
            model.eval()
            optimizer.zero_grad()
            pred = model(graph_batch)
            val_loss = criterion(pred, graph_batch.label)
            val_loss_accumulator.append(val_loss.cpu().item())
            val_pred_accumulator+=th.argmax(pred.cpu(), dim=-1).tolist()
            val_label_accumulator+=graph_batch.label.cpu().tolist()
        val_f1 = f1_score(val_label_accumulator, val_pred_accumulator, average='micro')
        mean_val_loss = np.mean(val_loss_accumulator)
        

        if epoch % period == 0:
            path2model_weights = path.join(path2checkpoints, f'checkpoint_{epoch:03d}.th')
            th.save(model.cpu(), path2model_weights)
            logger.success(f'a snapshot was saved at {path2model_weights}')
            model.to(device)

    path2model_weights = path.join(path2checkpoints, 'checkpoint_###.th')
    th.save(model.cpu(), path2model_weights)
    logger.success(f'training end | weights are saved at {path2model_weights}')

@router.command()
@click.option('--graph_filename', help='', type=str)
@click.option('--nb_epochs', type=int, default=32)
@click.option('--batch_size', type=int, default=16)
@click.option('--checkpoint', type=str)
@click.option('--period', default=8, type=int)
@click.option('--print_period', default=32, type=int)
@click.pass_context
def train_lightning(ctx, graph_filename, nb_epochs, batch_size, checkpoint, period, print_period):
    
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    logger.debug(f'target device : {device}')

    path2features = ctx.obj['path2features']
    path2checkpoints = ctx.obj['path2checkpoints']

    path2graph_filename = path.join(path2features, graph_filename)
    
    train_dataset = VDSDataset("/home/huthomas/Research/graph_relation/features/graphs.pkl", begin_percentage=0.0, end_percentage=0.75)
    val_dataset = VDSDataset("/home/huthomas/Research/graph_relation/features/graphs.pkl", begin_percentage=0.75, end_percentage=100.0)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    
    logger.debug('dataset and dataloader were initialized')

    path2initial_weights = path.join(path2checkpoints, checkpoint)
    if path.isfile(path2initial_weights):
        logger.debug(f'initial weights were detected, the model will use {path2initial_weights}')
        model = th.load(path2initial_weights, map_location=device)
    else:
        logger.debug('no initial weights were detected, the model will be created from scratch')
        model = SimpleGCN(train_dataset)
        model.to(device)

    print(model)
    logger.debug('model was initialized')

    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    logger.debug('training will start in 1 s')
    
    nb_graphs = len(train_dataset)
    mean_val_loss = 100.0
    val_f1=0.0
    for epoch in range(nb_epochs):
        cursor = 0
        for graph_batch in train_dataloader:
            crr_bt_size = graph_batch.label.shape[0]
            cursor += crr_bt_size
            model.train()
            optimizer.zero_grad()
            pred = model(graph_batch)
            loss = criterion(pred, graph_batch.label)
            loss.backward()
            optimizer.step()
            if int(cursor/batch_size)%print_period == 0:
                logger.debug(f'[{epoch:03d}/{nb_epochs:03d}] [{cursor:05d}/{nb_graphs:05d}] >> Loss: train {loss:07.3f} | val {mean_val_loss:07.3f} >> F1: {val_f1:07.3f}')
        val_loss_accumulator = []
        val_pred_accumulator, val_label_accumulator = [], []
        for graph_batch in val_dataloader:
            model.eval()
            optimizer.zero_grad()
            pred = model(graph_batch)
            val_loss = criterion(pred, graph_batch.label)
            val_loss_accumulator.append(val_loss.cpu().item())
            val_pred_accumulator+=th.argmax(pred.cpu(), dim=-1).tolist()
            val_label_accumulator+=graph_batch.label.cpu().tolist()
        val_f1 = f1_score(val_label_accumulator, val_pred_accumulator, average='micro')
        mean_val_loss = np.mean(val_loss_accumulator)
        

        if epoch % period == 0:
            path2model_weights = path.join(path2checkpoints, f'checkpoint_{epoch:03d}.th')
            th.save(model.cpu(), path2model_weights)
            logger.success(f'a snapshot was saved at {path2model_weights}')
            model.to(device)

    path2model_weights = path.join(path2checkpoints, 'checkpoint_###.th')
    th.save(model.cpu(), path2model_weights)
    logger.success(f'training end | weights are saved at {path2model_weights}')

if __name__ == '__main__':
    router(obj={})
