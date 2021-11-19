# import config
import argparse
from numpy.random.mtrand import rand
import torch
import numpy as np
from torch._C import device
from autoencoder import ConvDecoder, ConvEncoder
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
from lshash.lshash import LSHash
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
from pathlib import Path
# %matplotlib inline

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=3, type=int, help='number of total epochs')
parser.add_argument('--rootdir', default='/Users/abhishekvaidyanathan/Downloads/geological_similarity/', type=str, help='root directory for images')
parser.add_argument('--testImagePath',default="/Users/abhishekvaidyanathan/Downloads/geological_similarity/schist/ZZ5Z5.jpg",type=str,help='test image path')
parser.add_argument('--numImages',default=10,type=int,help='geological encoding')
parser.add_argument('--encoderModelPath',default="geological_encoding.pt",type=str,help='encoding model path')
parser.add_argument('--embeddingPath',default="geological_embed.npy",type=str,help='embedding path')
parser.add_argument('--kNearest',default=10,type=int,help='k nearest')
parser.add_argument('--resultFilePath',default="./results.csv",type=str,help='file path to save results.')
parser.add_argument('--nbitsList', nargs='+', type=int,default = [2, 4, 6, 8, 10, 12], help='nbits list')
parser.add_argument('--encoder',default='convencoder',type=str,help='encoder model')
args = parser.parse_args()

if args.encoder=='convencoder':
    encoder = ConvEncoder()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_image_files(rootdir):
    rootdir = rootdir
    image_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            image_files.append(os.path.join(subdir, file))
    return image_files

def load_tensor(image_path, device):
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def get_image_embedding(image_tensor):
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    return flattened_embedding

def get_image_embedding_array(image_files,encoder_model_path,embedding_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embedding_file = Path(embedding_path)
    encoder_file = Path(encoder_model_path)
    # encoder = ConvEncoder()
    if embedding_file.is_file() and encoder_file.is_file():
        image_embedding_array = []
        for images in image_files:
                image_tensor = load_tensor(images,device)
                # with torch.no_grad():
                #         image_embedding = encoder(image_tensor).cpu().detach().numpy()
                # flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
                flattened_embedding = get_image_embedding(image_tensor)
                image_embedding_array.append(flattened_embedding[0])
        return image_embedding_array
    encoder.load_state_dict(torch.load(encoder_model_path, map_location=device))
    encoder.eval()
    encoder.to(device)

    # Loads the embedding
    embedding = np.load(embedding_path)
    return embedding

def all_binary(n):
    total = 1 << n
    print(f"{total} possible combinations")
    combinations = []
    for i in range(total):
        # get binary representation of integer
        b = bin(i)[2:]
        # pad zeros to start of binary representtion
        b = '0' * (n - len(b)) + b
        b = [int(i) for i in b]
        combinations.append(b)
    return combinations

def randomProjection(nbits,image_embedding_array):
    class RandomProjection:
        # initialize what will be the buckets
        buckets = {}
        # initialize counter
        counter = 0

        def __init__(self, nbits, d):
            self.nbits = nbits
            self.d = d
            # create our hyperplane normal vecs for splitting data
            self.plane_norms = np.random.rand(d, nbits) - .5
            print(f"Initialized {self.plane_norms.shape[1]} hyperplane normal vectors.")
            # add every possible combination to hashes attribute as numpy array
            self.hashes = all_binary(nbits)
            # and add each as a key to the buckets dictionary
            for hash_code in self.hashes:
                # convert to string
                hash_code = ''.join([str(i) for i in hash_code])
                self.buckets[hash_code] = []
            # convert self.hashes to numpy array
            self.hashes = np.stack(self.hashes)

        def get_binary(self, vec):
            # calculate nbits dot product values
            direction = np.dot(vec, projection.plane_norms)
            # find positive direction (>0) and negative direction (<=0)
            direction = direction > 0
            # convert boolean array to integer strings
            binary_hash = direction.astype(int)
            return binary_hash
            
        def hash_vec(self, vec, show=False):
            # generate hash
            binary_hash = self.get_binary(vec)
            # convert to string format for dictionary
            binary_hash = ''.join(binary_hash.astype(str))
            # add ID to buckets dictionary
            self.buckets[binary_hash].append(self.counter)
            if show:
                print(f"{self.counter}: {''.join(binary_hash)}")
            # increment counter
            self.counter += 1
        
        def hamming(self, hashed_vec):
            # get hamming distance between query vec and all buckets in self.hashes
            hamming_dist = np.count_nonzero(hashed_vec != projection.hashes, axis=1).reshape(-1, 1)
            # add hash values to each row
            hamming_dist = np.concatenate((projection.hashes, hamming_dist), axis=1)
            # sort based on distance
            hamming_dist = hamming_dist[hamming_dist[:, -1].argsort()]
            return hamming_dist
        
        def top_k(self, vec, k=5):
            # generate hash
            binary_hash = self.get_binary(vec)
            # calculate hamming distance between all vectors
            hamming_dist = self.hamming(binary_hash)
            # loop through each bucket until we have k or more vector IDs
            vec_ids = []
            for row in hamming_dist:
                str_hash = ''.join(row[:-1].astype(str))
                bucket_ids = self.buckets[str_hash]
                vec_ids.extend(bucket_ids)
                if len(vec_ids) >= k:
                    vec_ids = vec_ids[:k]
                    break
            # return top k IDs
            return vec_ids
    projection = RandomProjection(nbits, np.asarray(image_embedding_array).shape[1])
    return projection

def get_top_k_similar_images(test_image,k,projection):
    top_k = projection.top_k(test_image, k)
    print("top k length: ",len(top_k))
    return top_k

def get_similar_images(image_embedding_array,top_k):
    similar_images = []
    for i in top_k:
        similar_images.append(image_embedding_array[i])
    return similar_images

def get_cosine_similarity(similar_images,comparison_image):
    cos = cosine_similarity(np.asarray(similar_images), [comparison_image])
    return np.mean(cos)

def get_random_test_images(image_embedding_array):
    random_images = []
    for i in range(10):
        random_image = random.choice(image_embedding_array)
        random_images.append(random_image)
    return random_images

def get_similarity(test_images,image_embedding_array,k,projection):
    results = {'xq': [], 'wb': []}
    image_tensor = load_tensor(test_images,device)
    flattened_embedding = get_image_embedding(image_tensor)
    test = flattened_embedding[0]
    # for test in test_images:
    top_k = projection.top_k(test, k)
    considered_images = []
    for i in top_k:
        considered_images.append(image_embedding_array[i])
    cos = cosine_similarity(considered_images, [test])
    cos = np.mean(cos)
    results['xq'].append(cos)
    cos = cosine_similarity(image_embedding_array, [test])
    cos = np.mean(cos)    
    results['wb'].append(cos)
    print(f"random images: {np.mean(results['xq'])}")
    print(f"all images: {np.mean(results['wb'])}")
    return results

def testing_func(image_embedding_array,test_images,k):
    testing = pd.DataFrame({
        'nbits': [],
        'random_images_sim': []
    })

    num_vecs = 10

    for epoch in range(args.epochs):
        print("------------printing results for epoch"+str(epoch)+"------------")
        for nbits in args.nbitsList:
            print("----------printing results for nbits:"+str(nbits)+"---------------")
            # initialize projection object
            projection = randomProjection(nbits,image_embedding_array)
            # add all our vectors
            for i in range(len(image_embedding_array)-1):
                projection.hash_vec(image_embedding_array[i])
            # get results from sim_check
            results = get_similarity(test_images,image_embedding_array,k,projection)
            testing = testing.append(pd.DataFrame({
                'epochs' : epoch,
                'nbits': nbits,
                'random_images_sim': results['xq']
            }), ignore_index=True)
            print("----------------------------------------------------------------")
        print("------------------------------------------------------------")
    testing.to_csv(args.resultFilePath,index=False)
    return testing

def main():
    rootdir = args.rootdir
    TEST_IMAGE_PATH = args.testImagePath
    NUM_IMAGES = args.numImages
    ENCODER_MODEL_PATH = args.encoderModelPath
    EMBEDDING_PATH = args.embeddingPath
    k = args.kNearest

    image_files = get_image_files(rootdir)
    image_embedding_array = get_image_embedding_array(image_files,ENCODER_MODEL_PATH,EMBEDDING_PATH)
    test_images = get_random_test_images(image_embedding_array)
    testing_df = testing_func(image_embedding_array,TEST_IMAGE_PATH,k)
    return testing_df

if __name__ == "__main__":
    testing_df = main()
    print("---------------------testing dataframe------------------------")
    print(testing_df)
    print("--------------------------------------------------------------")