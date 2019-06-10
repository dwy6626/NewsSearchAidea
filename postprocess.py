import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sorted_ans_path', type=str)
parser.add_argument('--ans_path', type=str)
args = parser.parse_args()

preds = np.load('preds.npy')

preds = preds.reshape(-1, 20)

candidate = np.loadtxt(args.ans_path, dtype=str, delimiter=',')

for query in range(20):
    for i in range(preds.shape[0]):
        if preds[i,query]==0:
            candidate[query+1] = np.concatenate((candidate[query+1, :i+1], candidate[query+1, i+2], candidate[query+1, i+1]), axis=1)


np.savetxt(args.sorted_ans_path, candidate[:, :301], delimiter=',', fmt='%s')