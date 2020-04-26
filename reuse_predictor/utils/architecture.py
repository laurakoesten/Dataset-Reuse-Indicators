import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class multi_RNN_with_additional_features(nn.Module):

    def __init__(self, params, init_embed_matrix=None):
        super(multi_RNN_with_additional_features, self).__init__()

        if params['use_description']:
            logging.info('Initialising Description pipeline...')
            self.text_embedding = nn.Embedding(params['num_included_tokens'] + 1, params['embed_dim'], padding_idx=0)
            if init_embed_matrix is not None:
                self.text_embedding.load_state_dict({'weight': torch.from_numpy(init_embed_matrix)})
                logging.info('Pre-trained embedding have been loaded successfully.')
            if params['embed_trainable']:
                logging.info('Embeddings will be fine-tuned further during training.')
                self.text_embedding.weight.requires_grad = True

            self.description_GRU = nn.GRU(params['embed_dim'], params['rnn_size'], params['num_layers'],
                                          batch_first=True, dropout=params['dropout'],
                                          bidirectional=params['is_bidirectional'], bias=True)
            if params['is_bidirectional']:
                self.description_embedding = nn.Linear(2 * params['rnn_size'], params['rnn_size'], bias=False)
            else:
                self.description_embedding = nn.Linear(params['rnn_size'], params['rnn_size'], bias=False)
            self.description_embedding2bn = nn.BatchNorm1d(params['rnn_size'])
            self.description_drop = nn.Dropout(params['dropout'])
        else:
            self.text_embedding = None
            self.description_GRU = None
            self.description_embedding = None
            self.description_embedding2bn = None
            self.description_drop = None

        if params['use_readme_features']:
            logging.info('Initialising README pipeline...')

            self.readme_embedding = nn.Linear(params['readme_features_dim'], params['rnn_size'], bias=False)
            self.readme_embedding2bn = nn.BatchNorm1d(params['rnn_size'])
            self.readme_drop = nn.Dropout(params['dropout'])
        else:
            self.readme_embedding = None
            self.readme_embedding2bn = None
            self.readme_drop = None

        if params['use_repo_features']:
            logging.info('Initialising Repository pipeline...')
            self.license_embedding = nn.Embedding(params['num_included_licenses'] + 1, params['rnn_size'])
            logging.info('Including License pipeline...')
            self.license_embedding2bn = nn.BatchNorm1d(params['rnn_size'])
            self.license_drop = nn.Dropout(params['dropout'])

            self.repo_embedding = nn.Linear(params['repo_features_dim'], params['rnn_size'], bias=False)
            self.repo_embedding2bn = nn.BatchNorm1d(params['rnn_size'])
            self.repo_drop = nn.Dropout(params['dropout'])
        else:
            self.license_embedding = None
            self.license_embedding2bn = None
            self.license_drop = None
            self.repo_embedding = None
            self.repo_embedding2bn = None
            self.repo_drop = None

        if params['use_data_files_features']:
            logging.info('Initialising Data Files pipeline...')
            self.data_files_GRU = nn.GRU(params['data_files_embed_dim'],
                                         params['rnn_size'],
                                         params['num_layers'],
                                         batch_first=True,
                                         dropout=params['dropout'],
                                         bidirectional=params['is_bidirectional'],
                                         bias=True)
            if params['is_bidirectional']:
                self.data_files_embedding = nn.Linear(2 * params['rnn_size'], params['rnn_size'], bias=False)
            else:
                self.data_files_embedding = nn.Linear(params['rnn_size'], params['rnn_size'], bias=False)

            self.data_files_embedding2bn = nn.BatchNorm1d(params['rnn_size'])
            self.data_files_drop = nn.Dropout(params['dropout'])
        else:

            self.data_files_GRU = None
            self.data_files_embedding = None
            self.data_files_embedding2bn = None
            self.data_files_drop = None

        self.h1 = nn.Linear(params['rnn_size'], params['rnn_size'], bias=False)
        self.h12bn = nn.BatchNorm1d(params['rnn_size'])
        self.h12drop = nn.Dropout(params['dropout'])
        self.h2 = nn.Linear(params['rnn_size'], params['rnn_size'], bias=False)
        self.h22bn = nn.BatchNorm1d(params['rnn_size'])
        self.h22drop = nn.Dropout(params['dropout'])
        self.h2y = nn.Linear(params['rnn_size'], params['num_classes'])

    def forward(self, input_descriptions=None, input_descriptions_lengths=None,
                input_data_files=None, input_data_files_lengths=None,
                input_readme_features=None, input_repo_features=None):

        if self.description_embedding:
            input_descriptions_lengths, description_sorted_idx = input_descriptions_lengths.sort(0, descending=True)

            description_embed_output = self.text_embedding(input_descriptions[description_sorted_idx])

            description_GRU_packed_output, description_h_T = self.description_GRU(
                torch.nn.utils.rnn.pack_padded_sequence(
                    description_embed_output,
                    input_descriptions_lengths,
                    batch_first=True))

            _, description_unsorted_idx = description_sorted_idx.sort(0)

            description_unsorted_h_T = description_h_T[:, description_unsorted_idx]

            if self.description_GRU.bidirectional:

                description_embedding = self.description_drop(self.description_embedding(
                    torch.cat((description_unsorted_h_T[-2], description_unsorted_h_T[-1]), dim=1)))
            else:

                description_embedding = self.description_drop(self.description_embedding(description_unsorted_h_T[-1]))

        if self.data_files_embedding:
            input_data_files_lengths, data_files_sorted_idx = input_data_files_lengths.sort(0, descending=True)
            data_files_GRU_packed_output, data_files_h_T = self.data_files_GRU(
                torch.nn.utils.rnn.pack_padded_sequence(input_data_files[data_files_sorted_idx],
                                                        input_data_files_lengths, batch_first=True))

            _, data_files_unsorted_idx = data_files_sorted_idx.sort(0)

            data_files_unsorted_h_T = data_files_h_T[:, data_files_unsorted_idx]

            if self.data_files_GRU.bidirectional:

                data_files_embedding = self.data_files_drop(self.data_files_embedding(
                    torch.cat((data_files_unsorted_h_T[-2], data_files_unsorted_h_T[-1]), dim=1)))
            else:

                data_files_embedding = self.data_files_drop(self.data_files_embedding(data_files_unsorted_h_T[-1]))

        if self.description_embedding and self.data_files_embedding and self.readme_embedding and self.repo_embedding:
            assert self.license_embedding is not None
            repo_embedding = self.repo_drop(
                self.repo_embedding2bn(self.repo_embedding(torch.cat(tuple(input_repo_features[1:]), dim=1))))
            lisense_embedding = self.license_drop(
                self.license_embedding2bn(self.license_embedding(input_repo_features[0])))
            readme_embedding = self.readme_drop(
                self.readme_embedding2bn(self.readme_embedding(torch.cat(tuple(input_readme_features), dim=1))))
            accumulated_out = nn.functional.relu(
                lisense_embedding + repo_embedding + description_embedding + readme_embedding + data_files_embedding)
        elif self.description_embedding and self.data_files_embedding and self.readme_embedding and self.repo_embedding is None:
            assert self.license_embedding is None
            readme_embedding = self.readme_drop(
                self.readme_embedding2bn(self.readme_embedding(torch.cat(tuple(input_readme_features), dim=1))))
            accumulated_out = nn.functional.relu(description_embedding + readme_embedding + data_files_embedding)
        elif self.description_embedding and self.data_files_embedding and self.readme_embedding is None and self.repo_embedding is None:
            assert self.license_embedding is None
            accumulated_out = nn.functional.relu(description_embedding + data_files_embedding)
        elif self.description_embedding and self.data_files_embedding and self.readme_embedding is None and self.repo_embedding:
            assert self.license_embedding is not None
            repo_embedding = self.repo_drop(
                self.repo_embedding2bn(self.repo_embedding(torch.cat(tuple(input_repo_features[1:]), dim=1))))
            lisense_embedding = self.license_drop(
                self.license_embedding2bn(self.license_embedding(input_repo_features[0])))
            accumulated_out = nn.functional.relu(
                lisense_embedding + repo_embedding + description_embedding + data_files_embedding)
        elif self.description_embedding and self.data_files_embedding is None and self.readme_embedding and self.repo_embedding:
            assert self.license_embedding is not None
            repo_embedding = self.repo_drop(
                self.repo_embedding2bn(self.repo_embedding(torch.cat(tuple(input_repo_features[1:]), dim=1))))
            lisense_embedding = self.license_drop(
                self.license_embedding2bn(self.license_embedding(input_repo_features[0])))
            readme_embedding = self.readme_drop(
                self.readme_embedding2bn(self.readme_embedding(torch.cat(tuple(input_readme_features), dim=1))))
            accumulated_out = nn.functional.relu(
                lisense_embedding + repo_embedding + description_embedding + readme_embedding)

        elif self.description_embedding and self.data_files_embedding is None and self.readme_embedding is None and self.repo_embedding:
            assert self.license_embedding is not None
            repo_embedding = self.repo_drop(
                self.repo_embedding2bn(self.repo_embedding(torch.cat(tuple(input_repo_features[1:]), dim=1))))
            lisense_embedding = self.license_drop(
                self.license_embedding2bn(self.license_embedding(input_repo_features[0])))
            accumulated_out = nn.functional.relu(lisense_embedding + repo_embedding + description_embedding)

        elif self.description_embedding and self.data_files_embedding is None and self.readme_embedding is None and self.repo_embedding is None:
            assert self.license_embedding is None
            accumulated_out = nn.functional.relu(description_embedding)

        elif self.description_embedding is None and self.data_files_embedding is None and self.readme_embedding is None and self.repo_embedding:
            repo_embedding = self.repo_drop(
                self.repo_embedding2bn(self.repo_embedding(torch.cat(tuple(input_repo_features[1:]), dim=1))))
            lisense_embedding = self.license_drop(
                self.license_embedding2bn(self.license_embedding(input_repo_features[0])))
            accumulated_out = nn.functional.relu(lisense_embedding + repo_embedding)

        elif self.description_embedding is None and self.data_files_embedding and self.readme_embedding is None and self.repo_embedding is None:
            assert self.license_embedding is None
            accumulated_out = nn.functional.relu(data_files_embedding)

        elif self.description_embedding is None and self.data_files_embedding is None and self.readme_embedding and self.repo_embedding is None:
            assert self.license_embedding is None
            readme_embedding = self.readme_drop(
                self.readme_embedding2bn(self.readme_embedding(torch.cat(tuple(input_readme_features), dim=1))))
            accumulated_out = nn.functional.relu(readme_embedding)
        h1 = nn.functional.relu(self.h12drop(self.h12bn(self.h1(accumulated_out))))
        h2 = nn.functional.relu(self.h22drop(self.h22bn(self.h2(h1))))

        h2y = self.h2y(h2)
        pred = nn.functional.log_softmax(h2y, dim=1)

        return pred
