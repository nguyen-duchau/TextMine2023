import click
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from os import getenv 

from rich.progress import track 
from libraries.log import logger 
from libraries.strategies import *
from data_holder import GraphDataset 
from modelization import TextMineGCN

from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
        with open(path.join(path2dataset, 'label2id.json'), 'r') as f:
            label2id = json.load(f)

        global_graph_accumulator = []
        documents = corpus['text'].tolist()
        labels = corpus['annotations'].tolist()
        ids = corpus['identifier'].tolist()
        # For each document in corpus
        for doc_id, doc in enumerate(track(documents, f'processing graphs...')):
            toks, token_pos, token_line = align_tokens(doc, labels[doc_id])
            tokens = tokenizer(toks, return_tensors='pt', return_offsets_mapping=True, is_split_into_words=True)
            tokens, offsets_mapping = {k:v for k,v in tokens.items() if k != 'offset_mapping'}, tokens['offset_mapping']

            embeddings_matrix = embedding_model(tokens['input_ids']).cpu()

            graph = build_graph(doc, embeddings_matrix[0,:,:], 2, labels[doc_id], ids[doc_id], offsets_mapping, token_line, token_pos, label2id)
            global_graph_accumulator.append(graph)
        serialize(path2_graphs, global_graph_accumulator, pickle)
        logger.success(f'{len(global_graph_accumulator):03d} dgl graphs were saved...!')
    except Exception as e:
        logger.error(e)
    
@router.command()
@click.option('--graph_filename', help='', type=str)
@click.option('--valgraph_filename', help='', type=str)
@click.option('--nb_epochs', type=int, default=32)
@click.option('--batch_size', type=int, default=16)
@click.option('--checkpoint', type=str)
@click.option('--period', default=8, type=int)
@click.pass_context
def train(ctx, graph_filename, valgraph_filename, nb_epochs, batch_size, checkpoint, period):
    
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    logger.debug(f'target device : {device}')

    path2features = ctx.obj['path2features']
    path2checkpoints = ctx.obj['path2checkpoints']

    path2graph_filename = path.join(path2features, graph_filename)
    path2valgraph_filename = path.join(path2features, valgraph_filename)
    
    train_dataset = GraphDataset(path2graph_filename, begin_percentage=0.0, end_percentage=0.75)
    val_dataset = GraphDataset(path2valgraph_filename, begin_percentage=0.75, end_percentage=100.0)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8)
    
    logger.debug('dataset and dataloader were initialized')

    path2initial_weights = path.join(path2checkpoints, checkpoint)
    if path.isfile(path2initial_weights):
        logger.debug(f'initial weights were detected, the model will use {path2initial_weights}')
        model = TextMineGCN.load_from_checkpoint(path2initial_weights, dataset=train_dataset)
    else:
        logger.debug('no initial weights were detected, the model will be created from scratch')
        model = TextMineGCN(train_dataset)
        model.to(device)

    print(model)
    logger.debug('model was initialized')

    logger.debug('training will start in 1 s')
    
    training_logger = TensorBoardLogger(path2initial_weights, name="tb_logs", version=0)

    print(model)
    logger.debug('model was initialized')
    callback = ModelCheckpoint(dirpath=training_logger.log_dir, save_top_k=3, monitor='val_loss', mode='min')
    trainer = pl.Trainer(max_epochs=nb_epochs, logger=training_logger, enable_checkpointing=True, default_root_dir=training_logger.log_dir, track_grad_norm=-1)
    logger.debug('training will start in 1 s')

    trainer.fit(model, train_dataloader, val_dataloader)
    
    path2model_weights = path.join(path2checkpoints, 'checkpoint_###.th')
    th.save(model.cpu(), path2model_weights)
    logger.success(f'training end | weights are saved at {path2model_weights}')

@router.command()
@click.option('--graph_filename', help='', type=str)
@click.option('--batch_size', type=int, default=16)
@click.option('--checkpoint', type=str)
@click.pass_context
def predict(ctx, graph_filename, batch_size, checkpoint):
    
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    logger.debug(f'target device : {device}')

    path2features = ctx.obj['path2features']
    path2checkpoints = ctx.obj['path2checkpoints']

    path2graph_filename = path.join(path2features, graph_filename)
    
    dataset = GraphDataset(path2graph_filename, begin_percentage=0.0, end_percentage=0.75)
    dataloader = DataLoader(dataset, batch_size, shuffle = True, num_workers=8)
    
    logger.debug('dataset and dataloader were initialized')

    path2initial_weights = path.join(path2checkpoints, checkpoint)
    model = TextMineGCN.load_from_checkpoint(path2initial_weights, dataset=dataset)

    print(model)
    logger.debug('model was initialized')

    for batch in dataloader:
        

if __name__ == '__main__':
    router(obj={})
