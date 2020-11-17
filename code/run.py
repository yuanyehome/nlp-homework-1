import argparse
import pickle
import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np


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
    return pickle.load(open(file_path, 'rb'))


class Weights:
    def __init__(self):
        self._weights = {}
        self._sum_weights = {}
        self._curr_steps = {}
        self._step = 0
        self._tmp_weights = None

    def final_update_sum(self):
        for key in self._weights.keys():
            self._sum_weights.setdefault(key, 0)
            self._sum_weights[key] += self._weights[key] * (self._step - self._curr_steps.get(key, 0))
            self._curr_steps[key] = self._step

    def update_weights(self, key, value):
        self._weights.setdefault(key, 0)
        self._sum_weights.setdefault(key, 0)
        self._sum_weights[key] += self._weights[key] * (self._step - self._curr_steps.get(key, 0))
        self._curr_steps[key] = self._step
        self._weights[key] += value

    def save_history(self):
        self._step += 1

    def get(self, key):
        self._weights.setdefault(key, 0)
        return self._weights[key]

    def prepare_predict_weights(self):
        self._tmp_weights = deepcopy(self._weights)
        self._weights = {key: self._sum_weights[key] / self._step for key in self._sum_weights.keys()}

    def reset_weights(self):
        self._weights = deepcopy(self._tmp_weights)
        self._tmp_weights = None


class Model:
    def __init__(self, trainset=None, weights=None):
        if not (trainset or weights):
            raise ValueError(
                "Error in model initialize: no trainset or weights provided!"
            )
        self.features, self.ys = None, None
        if trainset:
            (self.features, self.all_features), self.ys = \
                Model._get_features(trainset[0]), trainset[1]
            print("All features num: %d" % len(self.all_features))
            self.weights = Weights()
            assert len(self.features) == len(self.ys)
        if weights:
            self.weights = weights

    @staticmethod
    def _get_features(dataset, no_tqdm=False):
        all_features = []
        flaten_features = ['%d:%d' % (i, j)
                           for i in range(4) for j in range(4)]
        for x in tqdm(dataset, desc="Generating features", disable=no_tqdm):
            curr_features = []
            for i in range(len(x)):
                x_l2 = x[i - 2] if i >= 2 else '@'
                x_l1 = x[i - 1] if i >= 1 else '@'
                x_0 = x[i]
                x_r1 = x[i + 1] if i <= len(x) - 2 else '@'
                x_r2 = x[i + 2] if i <= len(x) - 3 else '@'
                _features = ['1' + x_0, '2' + x_l1, '3' + x_r1, '4' + x_l2 + x_l1,
                             '5' + x_l1 + x_0, '6' + x_0 + x_r1, '7' + x_r1 + x_r2]
                curr_features.append(_features)
                flaten_features += ['%s:%s' % (feature, str(label)) for feature in _features for label in range(4)]
            all_features.append(curr_features)
        return all_features, flaten_features

    def train(self, epoch_num):
        for i in range(epoch_num):
            for features, labels in tqdm(list(zip(self.features, self.ys)), desc="Training"):
                pred_labels = self._predict_one(features)
                if pred_labels != labels:
                    self._update_params(features, pred_labels, -1)
                    self._update_params(features, labels, 1)
                self.weights.save_history()
            self.weights.final_update_sum()

    def _update_params(self, features, labels, coef):
        for i, token_features in enumerate(features):
            for feature in token_features:
                self.weights.update_weights('%s:%s' % (feature, str(labels[i])), coef)
        for i in range(len(features) - 1):
            self.weights.update_weights(
                '%s:%s' % (labels[i], labels[i + 1]), coef)

    def predict(self, devset, out_path, labels, do_evaluation="no"):
        assert do_evaluation in ["no", "self", "perl"]
        self.weights.prepare_predict_weights()
        res = [self._decode(x) for x in tqdm(devset, desc="Predicting")]
        self.weights.reset_weights()
        with open(out_path, 'w') as f:
            f.write('\n'.join(res))
        if do_evaluation == "self":
            Model._evaluate(res, devset, labels)
        elif do_evaluation == "perl":
            os.system("./scripts/score scripts/word.txt %s %s > my_res/dev_score.txt" %
                      ("dev.txt", out_path))

    @staticmethod
    def _gen_words(sentence, labels):
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
        loc = 0
        res = []
        for word in word_list:
            res.append((loc, word))
            loc += len(word)
        return res

    @staticmethod
    def _evaluate(res, devset, labels):
        pred_words = [Model._gen_loc_words(word.split()) for word in res]
        refs_words = [Model._gen_loc_words(Model._gen_words(sentence, label_seq))
                      for sentence, label_seq in zip(devset, labels)]
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
        print("Dev num: %d; precision: %.3f; recall: %.3f; F1-score: %.3f" % (
            len(devset), precision, recall, f1
        ))

    def _decode(self, x):
        """
        Perhaps add more strict grammar?
        """
        features = self._get_features([x], no_tqdm=True)[0]
        labels = self._predict_one(features)
        word = ""
        words = []
        for token, label in zip(x, labels):
            word += token
            if label in [1, 3]:
                words.append(word)
                word = ""
        return ' '.join(words)

    def save_weights(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.weights, f)

    def _predict_one(self, x_features):
        transitions = [[self.weights.get('%s:%s' % (str(i), str(j))) for j in range(4)]
                       for i in range(4)]
        emissions = [[
            sum([self.weights.get('%s:%s' % (feature, str(label)))
                 for feature in token_features])
            for label in range(4)]
            for token_features in x_features]
        dp_matrix = [[[score, None] for score in emissions[0]]]
        for i in range(len(x_features) - 1):
            dp_matrix.append(
                [max([[dp_matrix[i][curr_label][0] + transitions[curr_label][next_label] + emissions[i + 1][next_label], curr_label]
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


def main():
    parser = argparse.ArgumentParser("A script for Chinese word cut.")
    parser.add_argument('--train_file', type=str,
                        help="The path to train file.")
    parser.add_argument('--predict_file', type=str,
                        required=True, help="The path to test file.")
    parser.add_argument('--output_path', type=str,
                        required=True, help="The path to the output file.")
    parser.add_argument('--weights', type=str,
                        help="The path for saving and loading the weights and features.")
    parser.add_argument('--epoch_num', type=int,
                        help="Epoch number of training.", default=5)
    args = parser.parse_args()

    if args.train_file:
        trainset = read_labeled_data(args.train_file)
        model = Model(trainset=trainset)
        model.train(epoch_num=args.epoch_num)
        model.save_weights(args.weights)
    else:
        assert args.weights is not None, "No train file or weights provided!"
        weights = load_weights(args.weights)
        model = Model(weights=weights)

    predict_data, labels = read_labeled_data(args.predict_file)
    model.predict(predict_data, args.output_path, labels, do_evaluation="self")


if __name__ == "__main__":
    main()
