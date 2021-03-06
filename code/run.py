import argparse
import pickle
import os
from tqdm import tqdm
from copy import deepcopy
from prettytable import PrettyTable
import numpy as np
np.random.seed(0)


def read_labeled_data(file_path):
    """
    B,E,M,S = 0,1,2,3
    """
    print('Reading data from %s...' % file_path)
    xs, ys = [], []
    empty_lines = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            words = line.strip().split()
            y = []
            for word in words:
                if len(word) == 1:
                    y.append(3)
                else:
                    y += [0] + [2 for _ in range(len(word) - 2)] + [1]
            x = ''.join(words)
            if x == "":
                empty_lines.append(i)
                continue
            assert len(x) == len(y)
            xs.append(x)
            ys.append(y)
    print('Done')
    print('Empty lines: %s' % empty_lines)
    print('Total lines: %d' % len(xs))
    return xs, ys


def load_weights(file_path):
    print('Loading weights from %s' % file_path)
    return pickle.load(open(file_path, 'rb'))


class Weights:
    def __init__(self, features, random_init):
        gauss_dist = np.random.normal(0, 1, len(features))
        if random_init:
            self._weights = {feature: number for feature,
                             number in zip(features, gauss_dist)}
        else:
            self._weights = {}
        self._sum_weights = {}
        self._curr_steps = {}
        self._step = 0
        self._tmp_weights = None

    def final_update_sum(self):
        for key in self._weights.keys():
            self._sum_weights.setdefault(key, 0)
            self._sum_weights[key] += self._weights[key] * \
                (self._step - self._curr_steps.get(key, 0))
            self._curr_steps[key] = self._step

    def update_weights(self, key, value):
        self._weights.setdefault(key, 0)
        self._sum_weights.setdefault(key, 0)
        self._sum_weights[key] += self._weights[key] * \
            (self._step - self._curr_steps.get(key, 0))
        self._curr_steps[key] = self._step
        self._weights[key] += value

    def save_history(self):
        self._step += 1

    def get(self, key):
        self._weights.setdefault(key, 0)
        return self._weights[key]

    def prepare_predict_weights(self):
        """Average the histories.
        """
        self._tmp_weights = deepcopy(self._weights)
        self._weights = {
            key: self._sum_weights[key] / self._step for key in self._sum_weights.keys()}

    def reset_weights(self):
        """Reset the weights to current weights to continue training.
        """
        self._weights = deepcopy(self._tmp_weights)
        self._tmp_weights = None


class Model:
    def __init__(self, feature_type, trainset=None, weights=None, validset=None, checkpoint=None, random_init=False):
        if not (trainset or weights):
            raise ValueError(
                "Error in model initialize: no trainset or weights provided!"
            )
        self.features, self.ys = None, None
        self.do_validation = False
        self.best_weight, self.best_score = None, 0
        self.checkpoint = checkpoint
        self.start_epoch = 0
        self.feature_type = feature_type
        if trainset:
            (self.features, self.all_features), self.ys = \
                Model._get_features(trainset[0], feature_type), trainset[1]
            print("Feature num: %d" % len(self.all_features))
            self.weights = Weights(self.all_features, random_init=random_init)
            assert len(self.features) == len(self.ys)
        if validset:
            self.valid_sentences, self.valid_labels = validset
            self.do_validation = True
        if weights:
            self.weights = weights
        if self.checkpoint and os.path.exists(self.checkpoint):
            ckpt = pickle.load(open(self.checkpoint, "rb"))
            self.weights, self.best_score, self.best_weight = \
                ckpt['model'], ckpt['best_score'], ckpt['best_model']
            self.start_epoch = ckpt['curr_epoch'] + 1
            print('Overridding feature type from checkpoint. Using %s' % ckpt['feature_type'])
            self.feature_type = ckpt['feature_type']

    @staticmethod
    def _gen_words(sentence, labels):
        """Generate the segmentation result from labels.
        """
        word = ""
        words = []
        for token, label in zip(sentence, labels):
            word += token
            if label in [1, 3]:
                words.append(word)
                word = ""
        return words

    @staticmethod
    def _gen_loc_words(word_list: list):
        """Generate a the location word pairs.
        """
        loc = 0
        res = []
        for word in word_list:
            res.append((loc, word))
            loc += len(word)
        return res

    @staticmethod
    def evaluate(res, sentences, labels):
        pred_words = [Model._gen_loc_words(word.split()) for word in res]
        refs_words = [Model._gen_loc_words(Model._gen_words(sentence, label_seq))
                      for sentence, label_seq in zip(sentences, labels)]
        correct, pred_num, refs_num = 0, 0, 0
        for pred, ref in zip(pred_words, refs_words):
            pred = set(pred)
            ref = set(ref)
            correct += len(pred & ref)
            pred_num += len(pred)
            refs_num += len(ref)
        precision = correct / pred_num
        recall = correct / refs_num
        f1 = 2 * precision * recall / (precision + recall)
        print("Dev num: %d; precision: %.4f; recall: %.4f; F1-score: %.4f" % (
            len(sentences), precision, recall, f1
        ))
        return precision, recall, f1

    @staticmethod
    def _get_features(dataset, feature_type, no_tqdm=False):
        sentence_features = []
        all_features = []
        for x in tqdm(dataset, desc="Generating features", disable=no_tqdm):
            curr_features = []
            for i in range(len(x)):
                if feature_type == "type1":
                    x_l2 = x[i - 2] if i >= 2 else '@'
                    x_l1 = x[i - 1] if i >= 1 else '@'
                    x_0 = x[i]
                    x_r1 = x[i + 1] if i <= len(x) - 2 else '@'
                    x_r2 = x[i + 2] if i <= len(x) - 3 else '@'
                    _features = ['1' + x_0, '2' + x_l1, '3' + x_r1, '4' + x_l2 + x_l1,
                                 '5' + x_l1 + x_0, '6' + x_0 + x_r1, '7' + x_r1 + x_r2]
                elif feature_type == "type2":
                    x_l1 = x[i - 1] if i >= 1 else '@'
                    x_0 = x[i]
                    x_r1 = x[i + 1] if i <= len(x) - 2 else '@'
                    _features = ['1' + x_0, '2' + x_l1, '3' + x_r1]
                elif feature_type == "type3":
                    x_l1 = x[i - 1] if i >= 1 else '@'
                    x_0 = x[i]
                    x_r1 = x[i + 1] if i <= len(x) - 2 else '@'
                    _features = ['1' + x_0, '2' + x_l1, '3' + x_r1,
                                 '4' + x_l1 + x_0, '5' + x_0 + x_r1]
                elif feature_type == "type4":
                    x_l3 = x[i - 3] if i >= 3 else '@'
                    x_l2 = x[i - 2] if i >= 2 else '@'
                    x_l1 = x[i - 1] if i >= 1 else '@'
                    x_0 = x[i]
                    x_r1 = x[i + 1] if i <= len(x) - 2 else '@'
                    x_r2 = x[i + 2] if i <= len(x) - 3 else '@'
                    x_r3 = x[i + 3] if i <= len(x) - 4 else '@'
                    _features = ['1' + x_0, '2' + x_l1, '3' + x_r1, '4' + x_l3 + x_l2, '5' + x_l2 + x_l1, 
                                 '6' + x_l1 + x_0, '7' + x_0 + x_r1, '8' + x_r1 + x_r2, '9' + x_r2 + x_r3,
                                 '10' + x_l3 + x_l2 + x_l1, '11' + x_l2 + x_l1 + x_0, '12' + x_l1 + x_0 + x_r1,
                                 '13' + x_0 + x_r1 + x_r2, '14' + x_r1 + x_r2 + x_r3]
                else:
                    raise ValueError
                curr_features.append(_features)
                all_features += ['%s:%s' % (feature, str(label))
                                 for feature in _features
                                 for label in range(4)]
            sentence_features.append(curr_features)
        return sentence_features, set(all_features)

    def _update_params(self, features, labels, coef):
        for i, token_features in enumerate(features):
            for feature in token_features:
                self.weights.update_weights(
                    '%s:%s' % (feature, str(labels[i])), coef)
        for i in range(len(features) - 1):
            self.weights.update_weights(
                '%s:%s' % (labels[i], labels[i + 1]), coef)

    def _check_grammar(seq, accepted_tokens):
        """Check the grammar of a given label sequence.
        """
        if len(seq) == 0:
            if accepted_tokens == [0, 3]:
                return True
            else:
                return False
        if seq[0] in accepted_tokens:
            curr_token = seq[0]
            if curr_token in [0, 2]:
                next_possible_tokens = [1, 2]
            elif curr_token in [1, 3]:
                next_possible_tokens = [0, 3]
            else:
                raise ValueError
            return Model._check_grammar(seq[1:], next_possible_tokens)
        return False

    def _decode(self, x):
        """Generate the label sequence of one sentence.
        """
        features = Model._get_features(
            [x], self.feature_type, no_tqdm=True)[0][0]
        labels = self._predict_one(features)
        correct_grammar = Model._check_grammar(labels, [0, 3])
        word = ""
        words = []
        for token, label in zip(x, labels):
            word += token
            if label in [1, 3]:
                words.append(word)
                word = ""
        return ' '.join(words), correct_grammar

    def _predict_one(self, x_features):
        """Predict the labels from the given features of a single sentence.
        """
        label_score = [[self.weights.get('%s:%s' % (str(i), str(j))) for j in range(4)]
                       for i in range(4)]  # The reward from edge features for each action.
        token_score = [[
            sum([self.weights.get('%s:%s' % (feature, str(label)))
                 for feature in token_features])
            for label in range(4)]
            for token_features in x_features]  # The reward from node features for each action.
        # Save the best path at each time step.
        dp_matrix = [[[score, None] for score in token_score[0]]]
        for i in range(len(x_features) - 1):
            dp_matrix.append(
                [max([[dp_matrix[i][curr_label][0] + label_score[curr_label][next_label] +
                       token_score[i + 1][next_label], curr_label]
                      for curr_label in range(4)
                      ])
                 for next_label in range(4)
                 ])
        max_value_choice_pair = max(
            [[dp_matrix[-1][choice], choice] for choice in range(4)])
        path = []
        length = len(x_features)
        for i in range(length):
            path.append(max_value_choice_pair[1])
            max_value_choice_pair = dp_matrix[length -
                                              1-i][max_value_choice_pair[1]]
        return list(reversed(path))

    def predict(self, sentences):
        self.weights.prepare_predict_weights()
        res = [self._decode(x) for x in tqdm(sentences, desc="Predicting")]
        grammar_correct_rate = sum([item[1] for item in res]) / len(res)
        res = [item[0] for item in res]
        self.weights.reset_weights()
        return res, grammar_correct_rate

    def save_weights(self, file_path):
        print('Best F1 score is %.4f' % self.best_score)
        with open(file_path, 'wb') as f:
            pickle.dump({
                'weight': self.best_weight,
                'feature_type': self.feature_type
            }, f)

    def train(self, epoch_num, log_file):
        self.table = PrettyTable(
            ['Epoch', 'Precision', 'Recall', 'F1-score', 'Grammar correct rate'])
        for i in range(self.start_epoch, self.start_epoch + epoch_num):
            for features, labels in tqdm(list(zip(self.features, self.ys)),
                                         desc="Training for epoch %d" % i):
                pred_labels = self._predict_one(features)
                if pred_labels != labels:
                    self._update_params(features, pred_labels, -1)
                    self._update_params(features, labels, 1)
                self.weights.save_history()
            self.weights.final_update_sum()
            if self.do_validation:
                dev_res, grammar_correct_rate = self.predict(
                    self.valid_sentences)
                precision, recall, f1 = Model.evaluate(dev_res, self.valid_sentences,
                                                       self.valid_labels)
                self.table.add_row(
                    [i, precision, recall, f1, grammar_correct_rate])
                print('Grammar correct rate in this epoch: %.6f' %
                      grammar_correct_rate)
                if f1 > self.best_score:
                    print('Best model from epoch %d.' % i)
                    self.best_score = f1
                    self.best_weight = deepcopy(self.weights)
            else:
                self.best_weight = self.weights
            if self.checkpoint:
                print('Saving checkpoint')
                pickle.dump({
                    "model": self.weights,
                    "best_score": self.best_score,
                    "best_model": self.best_weight,
                    "curr_epoch": i,
                    "feature_type": self.feature_type
                }, open(self.checkpoint, "wb"))
                print('Done')
        print(self.table, file=open(log_file, 'w'))


def main():
    parser = argparse.ArgumentParser("A script for Chinese word cut.")
    parser.add_argument('--train_file', type=str,
                        help="The path to train file.")
    parser.add_argument('--valid_file', type=str,
                        help='Validation data.')
    parser.add_argument('--predict_file', type=str,
                        help="The path to test file.")
    parser.add_argument('--output_path', type=str,
                        help="The path to the output file.")
    parser.add_argument('--weights', type=str,
                        help="The path for saving and loading the weights and features.")
    parser.add_argument('--epoch_num', type=int,
                        help="Epoch number of training.", default=50)
    parser.add_argument('--checkpoint', type=str,
                        help="Path for saving and load checkpoint")
    parser.add_argument('--log_file', type=str,
                        help="Path to log file.")
    parser.add_argument('--random_init', action="store_true",
                        help="Whether initilize parameters using Gaussian Distribution.")
    parser.add_argument('--feature_type', type=str, required=True, choices=['type1', 'type2', 'type3', 'type4'],
                        help="Choose what features to use.")
    args = parser.parse_args()

    if args.train_file:
        trainset = read_labeled_data(args.train_file)
        if args.valid_file:
            validset = read_labeled_data(args.valid_file)
        else:
            validset = None
        model = Model(trainset=trainset, validset=validset,
                      checkpoint=args.checkpoint, random_init=args.random_init, feature_type=args.feature_type)
        model.train(epoch_num=args.epoch_num, log_file=args.log_file)
        if args.weights:
            model.save_weights(args.weights)
    else:
        assert args.weights is not None, "No train file or weights provided!"
        weights = load_weights(args.weights)
        weights, args.feature_type = weights['weights'], weights['feature_type']
        print('Override the feature type using %s' % args.feature_type)
        model = Model(weights=weights, feature_type=args.feature_type)
        if args.valid_file:
            print('Validating for this model...')
            sentences, labels = read_labeled_data(args.valid_file)
            res, grammar_correct_rate = model.predict(sentences)
            Model.evaluate(res, sentences, labels)

    if args.predict_file:
        assert args.output_path, 'No prediction output specified!'
        predict_data, _ = read_labeled_data(args.predict_file)
        res, grammar_correct_rate = model.predict(predict_data)
        with open(args.output_path, 'w') as f:
            f.write('\n'.join(res))


if __name__ == "__main__":
    main()
