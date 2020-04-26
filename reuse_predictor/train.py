#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse
import cPickle as pickle
import json
from sklearn import metrics
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from utils.architecture import multi_RNN_with_additional_features
from utils.utilities import save_checkpoint

logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser(description='Training')
parser.add_argument("--use_readme_features",
                    action='store_true',
                    help='Include README features.')
parser.add_argument("--use_repo_features",
                    action='store_true',
                    help='Include general repository features.')
parser.add_argument("--use_description",
                    action='store_true',
                    help='Include description features.')
parser.add_argument("--use_data_files_features",
                    action='store_true',
                    help='Include data files features.')
args = parser.parse_args()
assert (args.use_data_files_features or
        args.use_repo_features or
        args.use_readme_features or
        args.use_description), ("You should initialise training with at least a set of features ", ".")

params = {'batch_size': 128,
          'num_layers': 2,
          'init_weight': 1e-2,
          'embed_trainable': True,
          'embed_dim': 300,
          'rnn_size': 512,
          'dropout': 0.0,
          'max_epoch': 50,
          'learning_rate': 1e-3,
          'weight_decay': 1e-2,
          'is_bidirectional': True,
          'use_repo_features': args.use_repo_features,
          'use_description': args.use_description,
          'use_readme_features': args.use_readme_features,
          'use_data_files_features': args.use_data_files_features,
          'device': torch.device("cuda:0")
          }



dataset_load_location = '../data/processed_dataset/'
glove_vectors_loc = '../data/glove/glove.1917494.300d.npy'
glove_vectors_dict_loc = '../data/glove/glove.1917494.300d.json'

checkpoint_dump_prefix = 'use_repo_features_%s.use_data_files_%s.use_readme_%s.use_description_%s' % (
    params['use_repo_features'],
    params['use_data_files_features'],
    params['use_readme_features'],
    params['use_description'])

with open(dataset_load_location + 'idx_to_class.pickle', 'rb') as f:
    idx_to_class = pickle.load(f)
params['num_classes'] = len(idx_to_class.keys())
class_weights = torch.FloatTensor([idx_to_class[c]['weight'] for c in idx_to_class])
logging.info('Num of Classes: %d' % params['num_classes'])
logging.info('Class Weights: %s' % ' '.join([str(weight.item()) for weight in class_weights]))

with open(dataset_load_location + 'descriptions_dictionary.json', 'r') as f:
    descriptions_dictionary = json.load(f, 'utf-8')
    id2word = descriptions_dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = descriptions_dictionary['word2id']
with open(dataset_load_location + 'licenses_dictionary.json', 'r') as f:
    licenses_dictionary = json.load(f, 'utf-8')
    id2license = licenses_dictionary['id2license']
    id2license = {int(key): id2license[key] for key in id2license}
    license2id = licenses_dictionary['license2id']

with open(dataset_load_location + 'train_batched_dataset.p', 'rb') as f:
    train_batched_dataset = pickle.load(f)
with open(dataset_load_location + 'train_dataset.p', 'rb') as f:
    train_dataset = pickle.load(f)

with open(dataset_load_location + 'val_batched_dataset.p', 'rb') as f:
    val_batched_dataset = pickle.load(f)
with open(dataset_load_location + 'val_dataset.p', 'rb') as f:
    val_dataset = pickle.load(f)

with open(dataset_load_location + 'test_batched_dataset.p', 'rb') as f:
    test_batched_dataset = pickle.load(f)
with open(dataset_load_location + 'test_dataset.p', 'rb') as f:
    test_dataset = pickle.load(f)


def assert_loaded_corpora(input_dataset):
    assert type(input_dataset) is dict or type(input_dataset) is list
    if type(input_dataset) is dict:
        for key in input_dataset:
            for j in range(0, len(input_dataset[key])):
                assert len(input_dataset[key][j]) == params['batch_size']
    else:
        for j in range(0, len(input_dataset)):
            assert len(input_dataset[j]) == params['batch_size']


assert_loaded_corpora(train_batched_dataset)
assert_loaded_corpora(train_dataset)
assert_loaded_corpora(val_batched_dataset)
assert_loaded_corpora(val_dataset)
assert_loaded_corpora(test_batched_dataset)
assert_loaded_corpora(test_dataset)

assert len(word2id.keys()) == len(id2word.keys())
assert len(license2id.keys()) == len(id2license.keys())
params['num_included_tokens'] = len(id2word.keys())
params['num_included_licenses'] = len(id2license.keys())

if params['use_readme_features']:
    readme_features = ['num_headers',
                       'num_code_blocks',
                       'num_images',
                       'is_english_in_ids',
                       'num_urls',
                       'num_tokens',
                       'num_tables',
                       'num_inline_code_blocks']
else:
    readme_features = []

if params['use_repo_features']:
    repo_features = ['license_in_ids',
                     'size',
                     'repo_age',
                     'open_to_total_issues_ratio',
                     'not_openable_data_files_ratio',
                     'data_files_ratio']
else:
    repo_features = []


def assert_readme_features_present(readme_features,
                                   input_dataset,
                                   input_batched_dataset):
    readme_features_input_dim = {}
    for item in readme_features:
        assert item in input_dataset
        for batched_item in input_dataset[item]:
            assert batched_item.size(0) == params['batch_size']
            if item not in readme_features_input_dim:
                # We are not expecting three-dimensional feature vectors for the READMEs.
                readme_features_input_dim[item] = 1 if batched_item.dim() == 1 else batched_item.size(1)
            else:
                if batched_item.dim() == 1:
                    assert readme_features_input_dim[item] == 1
                else:
                    assert readme_features_input_dim[item] == batched_item.size(1)

        for batch in input_batched_dataset:
            for instance in batch:
                assert item in instance['readme']
    return sum([readme_features_input_dim[item] for item in readme_features_input_dim])


def assert_repo_features_present(repo_features,
                                 input_dataset,
                                 input_batched_dataset):
    repo_features_input_dim = {}
    for item in repo_features:
        assert item in input_dataset
        for batched_item in input_dataset[item]:
            assert batched_item.size(0) == params['batch_size']
            if item not in repo_features_input_dim:
                # We are not expecting three-dimensional feature vectors for the READMEs.
                repo_features_input_dim[item] = 1 if batched_item.dim() == 1 else batched_item.size(1)
            else:
                if batched_item.dim() == 1:
                    assert repo_features_input_dim[item] == 1
                else:
                    assert repo_features_input_dim[item] == batched_item.size(1)

        for batch in input_batched_dataset:
            for instance in batch:
                assert item in instance
    # The minus below is due to the license_in_ids which is given to 
    # different Linear layer in the network.
    total_repo_features_dim = sum([repo_features_input_dim[item] for item in
                                   repo_features_input_dim]) - 1 if 'license_in_ids' in repo_features else sum(
        [repo_features_input_dim[item] for item in repo_features_input_dim])
    return total_repo_features_dim


def assert_data_files_features_dim(input_dataset):
    data_files_features_input_dim = {'max_data_files_per_repo': 0, 'data_files_embed_dim': 0}

    for batched_item in input_dataset['processed_data_files']:
        assert batched_item.size(0) == params['batch_size']
        if data_files_features_input_dim['max_data_files_per_repo'] == 0:
            # We are not expecting three-dimensional feature vectors for the READMEs.
            data_files_features_input_dim['max_data_files_per_repo'] = batched_item.size(1)
            data_files_features_input_dim['data_files_embed_dim'] = batched_item.size(2)
        else:
            assert data_files_features_input_dim['max_data_files_per_repo'] == batched_item.size(1)
            assert data_files_features_input_dim['data_files_embed_dim'] == batched_item.size(2)

    return data_files_features_input_dim


params['readme_features_dim'] = assert_readme_features_present(readme_features, train_dataset, train_batched_dataset)
assert assert_readme_features_present(readme_features, val_dataset, val_batched_dataset) == params[
    'readme_features_dim']
assert assert_readme_features_present(readme_features, test_dataset, test_batched_dataset) == params[
    'readme_features_dim']

params['repo_features_dim'] = assert_repo_features_present(repo_features, train_dataset, train_batched_dataset)
assert assert_repo_features_present(repo_features, val_dataset, val_batched_dataset) == params['repo_features_dim']
assert assert_repo_features_present(repo_features, test_dataset, test_batched_dataset) == params['repo_features_dim']

data_files_features_dims_ = assert_data_files_features_dim(train_dataset)
for key in data_files_features_dims_:
    assert key not in params
    params[key] = data_files_features_dims_[key]

data_files_features_dims_ = assert_data_files_features_dim(val_dataset)
for key in data_files_features_dims_:
    assert key in params
    assert params[key] == data_files_features_dims_[key]
data_files_features_dims_ = assert_data_files_features_dim(test_dataset)
for key in data_files_features_dims_:
    assert key in params
    assert params[key] == data_files_features_dims_[key]

logging.info(params)

# Loading Glove Vectors
with open(glove_vectors_loc, 'rb') as f:
    glove_vectors = np.load(f)
with open(glove_vectors_dict_loc, 'rb') as f:
    glove_vectors_dict = json.load(f, encoding='utf-8')

word2vec = {}
for word in glove_vectors_dict:
    word2vec[word] = glove_vectors[glove_vectors_dict[word]]

# We set num_words + 1 in case it's needed for the PADs.
init_embed_matrix = np.random.uniform(low=-params['init_weight'], high=params['init_weight'],
                                      size=(params['num_included_tokens'] + 1, params['embed_dim']))
init_embed_matrix[0] = torch.zeros(params['embed_dim'])

num_words_init_with_glove = 0

for word in word2id:
    if word in word2vec:
        init_embed_matrix[word2id[word]] = word2vec[word]
        num_words_init_with_glove += 1
logging.info('%d out of %d included tokens have been initialised with pre-trained embeddings.',
             num_words_init_with_glove,
             params['num_included_tokens'])


def weights_init(module):
    if type(module) == nn.GRU:
        for parameter in module.parameters():
            parameter.data.uniform_(-params['init_weight'], params['init_weight'])
    if type(module) == nn.Linear:
        module.weight.data.uniform_(-params['init_weight'], params['init_weight'])
        if module.bias is not None:
            module.bias.data.uniform_(-params['init_weight'], params['init_weight'])
    if type(module) == nn.BatchNorm1d:
        module.weight.data.fill_(1.0)
        if module.bias is not None:
            module.bias.data.zero_()


def assert_eval_state_on(module):
    assert module.training == False


model = multi_RNN_with_additional_features(params,
                                           init_embed_matrix=init_embed_matrix)

criterion = nn.NLLLoss(reduction='mean')
model.apply(weights_init)

# Transfer the module to the selected GPU or CPU.
model.to(params['device'])
criterion.to(params['device'])

optimiser = torch.optim.Adam(model.parameters(),
                             lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                       factor=5e-1,
                                                       patience=2,
                                                       mode='max',
                                                       verbose=True)


def evaluate(input_dataset, input_batched_dataset):
    model.eval()
    model.apply(assert_eval_state_on)
    assert len(input_batched_dataset) == len(input_dataset['processed_description_in_ids'])
    num_batches = len(input_dataset['processed_description_in_ids'])
    y_pred = []
    y_true = []

    for batch_idx in range(0, num_batches):
        if params['device'].type == 'cpu':
            pred = model(input_dataset['processed_description_in_ids'][batch_idx],
                         input_dataset['len_processed_description_in_ids'][batch_idx],
                         input_dataset['processed_data_files'][batch_idx],
                         input_dataset['len_processed_data_files'][batch_idx],
                         return_readme_input_features(input_dataset, batch_idx),
                         return_repo_input_features(input_dataset, batch_idx))

            batch_y_pred = pred.max(1)[1].numpy()
        else:
            pred = model(
                input_dataset['processed_description_in_ids'][batch_idx].cuda(params['device'], non_blocking=True),
                input_dataset['len_processed_description_in_ids'][batch_idx].cuda(params['device'], non_blocking=True),
                input_dataset['processed_data_files'][batch_idx].cuda(params['device'], non_blocking=True),
                input_dataset['len_processed_data_files'][batch_idx].cuda(params['device'], non_blocking=True),
                return_readme_input_features(input_dataset, batch_idx),
                return_repo_input_features(input_dataset, batch_idx))
            batch_y_pred = pred.max(1)[1].cpu().numpy()
        y_pred.append(batch_y_pred)

        batch_y_true = input_dataset['class'][batch_idx].numpy()
        y_true.append(batch_y_true)

    model.train()

    return metrics.accuracy_score(np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)), metrics.f1_score(
        np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0), average='macro')


def return_readme_input_features(input_dataset, batch_idx):
    readme_input_features_list = []
    for key in readme_features:
        if key != 'is_english_in_ids':
            if params['device'].type == 'cpu':
                readme_input_features_list.append(input_dataset[key][batch_idx].unsqueeze(1))
            else:
                readme_input_features_list.append(
                    input_dataset[key][batch_idx].unsqueeze(1).cuda(params['device'], non_blocking=True))
        else:
            if params['device'].type == 'cpu':
                readme_input_features_list.append(input_dataset[key][batch_idx])
            else:
                readme_input_features_list.append(
                    input_dataset[key][batch_idx].cuda(params['device'], non_blocking=True))
    return readme_input_features_list


def return_repo_input_features(input_dataset, batch_idx):
    repo_input_features_list = []
    for key in repo_features:
        if key != 'license_in_ids':
            if params['device'].type == 'cpu':
                repo_input_features_list.append(input_dataset[key][batch_idx].unsqueeze(1))
            else:
                repo_input_features_list.append(
                    input_dataset[key][batch_idx].unsqueeze(1).cuda(params['device'], non_blocking=True))
        else:
            if params['device'].type == 'cpu':
                repo_input_features_list.append(input_dataset[key][batch_idx])
            else:
                repo_input_features_list.append(input_dataset[key][batch_idx].cuda(params['device'], non_blocking=True))
    return repo_input_features_list


if __name__ == '__main__':
    model.train()
    train_val_test_results = {'train_acc': [], 'train_f1': [], 'val_acc': [], 'val_f1': [], 'test_acc': [],
                              'test_f1': []}
    max_val_acc = 0.0
    max_val_f1 = 0.0

    for epoch in range(0, params['max_epoch']):
        accumulated_err = 0.0
        start_time = time.time()
        for batch_idx in range(0, len(train_dataset['processed_description_in_ids'])):
            if params['device'].type == 'cpu':
                batched_readme_features = []
                pred = model(train_dataset['processed_description_in_ids'][batch_idx],
                             train_dataset['len_processed_description_in_ids'][batch_idx],
                             train_dataset['processed_data_files'][batch_idx],
                             train_dataset['len_processed_data_files'][batch_idx],
                             return_readme_input_features(train_dataset, batch_idx),
                             return_repo_input_features(train_dataset, batch_idx))
                batchLabels = train_dataset['class'][batch_idx]
                err = criterion(pred, batchLabels)
            else:
                pred = model(
                    train_dataset['processed_description_in_ids'][batch_idx].cuda(params['device'], non_blocking=True),
                    train_dataset['len_processed_description_in_ids'][batch_idx].cuda(params['device'],
                                                                                      non_blocking=True),
                    train_dataset['processed_data_files'][batch_idx].cuda(params['device'], non_blocking=True),
                    train_dataset['len_processed_data_files'][batch_idx].cuda(params['device'], non_blocking=True),
                    return_readme_input_features(train_dataset, batch_idx),
                    return_repo_input_features(train_dataset, batch_idx))
                batchLabels = train_dataset['class'][batch_idx].cuda(params['device'], non_blocking=True)

                err = criterion(pred, batchLabels.cuda(params['device'], non_blocking=True))

            accumulated_err += err

            print('%d / %d: Training Error: %.3f' % (batch_idx + 1,
                                                     len(train_dataset['processed_description_in_ids']),
                                                     err), end='\r')
            optimiser.zero_grad()
            err.backward()
            optimiser.step()
        print('')
        logging.info('%d out of %d Epochs: Accumulated Training Error Across All Batches : %.2f',
                     epoch,
                     params['max_epoch'],
                     accumulated_err)
        logging.info('Epoch completed in: %.2f seconds.', (time.time() - start_time))
        training_acc, training_f1 = evaluate(train_dataset, train_batched_dataset)
        validation_acc, validation_f1 = evaluate(val_dataset, val_batched_dataset)
        testing_acc, testing_f1 = evaluate(test_dataset, test_batched_dataset)
        train_val_test_results['train_acc'].append(training_acc)
        train_val_test_results['train_f1'].append(training_f1)
        train_val_test_results['val_acc'].append(validation_acc)
        train_val_test_results['val_f1'].append(validation_f1)
        train_val_test_results['test_acc'].append(testing_acc)
        train_val_test_results['test_f1'].append(testing_f1)
        logging.info('[TRAINING] F1: %.3f --- ACC: %.3f', training_f1, training_acc)
        logging.info('[VALIDATION] F1: %.3f --- ACC: %.3f', validation_f1, validation_acc)
        logging.info('[TESTING] F1: %.3f --- ACC: %.3f', testing_f1, testing_acc)

        # Re-computing the learning rate based on the selected scheduling method.
        scheduler.step(validation_f1)

        if validation_f1 > max_val_f1:
            state = {'epoch': epoch + 1,
                     'train_val_test_results': train_val_test_results,
                     'model': model.state_dict(),
                     'optim_dict': optimiser.state_dict(),
                     'params': params}

            save_checkpoint(state,
                            checkpoint_dir='./checkpoints',  # path to folder
                            checkpoint_prefix='%s.epoch_%d.val_acc_%.2f.test_acc_%.2f' % (
                                checkpoint_dump_prefix,
                                epoch + 1,
                                validation_acc,
                                testing_acc))

            max_val_f1 = validation_f1
        print('--------------------------------------------------------------------------')