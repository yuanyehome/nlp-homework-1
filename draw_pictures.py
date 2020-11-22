import matplotlib.pyplot as plt
import numpy as np


def read_logs(file_path):
    scores = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line[0] == '+':
                continue
            line = line.split('|')
            if line[1].strip() == 'Epoch':
                continue
            score = float(line[4].strip())
            scores.append(score)
    assert len(scores) == 50
    return scores


def main():
    logs_1 = read_logs('my_res/res_type_1/log.txt')
    logs_2 = read_logs('my_res/res_type_2/log.txt')
    logs_3 = read_logs('my_res/res_type_3/log.txt')
    logs_4 = read_logs('my_res/res_type_4/log.txt')
    logs_random = read_logs('my_res/res_version_2/log.txt')
    x = np.linspace(1, len(logs_1), 50)
    plt.plot(x, logs_1, label='type 1')
    plt.plot(x, logs_2, label='type 2')
    plt.plot(x, logs_3, label='type 3')
    plt.plot(x, logs_4, label='type 4')
    plt.plot(x, logs_random, label='type 5')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Vaslid F1-score')
    plt.savefig('compare_1.pdf')
    plt.clf()


if __name__ == "__main__":
    main()
