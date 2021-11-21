import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--testImagePath',default="/Users/abhishekvaidyanathan/Downloads/geological_similarity/schist/ZZ5Z5.jpg",type=str,help='test image path')
parser.add_argument('--numImages',default=10,type=int,help='geological encoding')
parser.add_argument('--encoderModelPath',default="./encoders/geological_encoding.pt",type=str,help='encoding model path')
parser.add_argument('--embeddingPath',default="./encoders/geological_embed.npy",type=str,help='embedding path')
parser.add_argument('--kNearest', nargs='+', type=int,default = [5, 10, 15, 20, 50, 100], help='k Nearest list')
parser.add_argument('--resultFilePath',default="./lsh-results/lsh-results.csv",type=str,help='file path to save results.')
parser.add_argument('--nbitsList', nargs='+', type=int,default = [2, 4, 6, 8, 10, 12], help='nbits list')
parser.add_argument('--encoder',default='convencoder',type=str,help='encoder model')
args = parser.parse_args()

def load_csv(filePath):
    dataset = pd.read_csv(filePath)
    return dataset

