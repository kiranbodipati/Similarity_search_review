'''
importing necessary libraries
'''
import torch
import numpy as np
from encoderPreprocess.autoencoder import ConvDecoder, ConvEncoder
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import random
import argparse


'''
setting a random seed to regenerate same results
'''
random.seed(10)


'''
Getting arguments from user input
'''
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--imagesFilePath',default="./image-files/geological_map.json",type=str,help='image path files')
parser.add_argument('--resultFilePath',default="./knn-results/results.csv",type=str,help='file path to save results.')
parser.add_argument('--resultFilePathJson',default="./knn-results/results.json",type=str,help='file path to save results.')
parser.add_argument('--kNearest',default=10,type=int,help='top k images')
parser.add_argument('--encoderModelPath',default="./encoders/geological_encoding.pt",type=str,help='encoding model path')
parser.add_argument('--embeddingPath',default="./encoders/geological_embed.npy",type=str,help='embedding path')

args = parser.parse_args()

TEST_IMAGE_PATH = "geological_similarity/andesite/0A5NL.jpg"
K_IMAGES = args.kNearest
ENCODER_MODEL_PATH = args.encoderModelPath
EMBEDDING_PATH = args.embeddingPath

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
encoder = ConvEncoder()
encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
encoder.eval()
encoder.to(device)
embedding = np.load(EMBEDDING_PATH)

def convert_image_to_tensor(image_path):
    '''
    Given an image file path, loads the image into a tensor and returns the image tensor
    '''
    tensor= T.ToTensor()(Image.open(image_path))
    tensor = tensor.unsqueeze(0)
    return tensor

def load_images():
    '''
    reads all the image file paths stored in ./image-files/geological_mapping_data.json or any other user inputted file path, and stores them into a list
    returns list of all image file paths
    '''
    image_paths=[]
    # with open("geological_map.json", 'r', encoding='utf-8') as f:
    #     image_paths=json.load(f)
    with open(args.imagesFilePath, 'r', encoding='utf-8') as f:
        image_paths=json.load(f)
    print(len(image_paths))
    return image_paths

def find_similar_images(image_path, k, embedding, device):
    '''
    finds the K nearest neighbours given an image path, embeddings, and K
    returns the indices list of the K nearest neighbours and the testing time taken
    '''
    image_tensor = convert_image_to_tensor(image_path, device)
    # image_tensor = image_tensor.to(device)


    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(embedding)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    #print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    #print(flattened_embedding.shape)

    start_time=time.time()
    _, indices = knn.kneighbors(flattened_embedding)
    end_time=time.time()
    time_taken=end_time-start_time
    #print("Time taken: ",end_time-start_time)
    indices_list = indices.tolist()
    
    #print(indices_list)
    return indices_list,time_taken

def display_similar_images(image_paths,indices_list):
    '''
    function to display the k most similar images using matplotlib
    '''
    indices = indices_list[0]
    print("total indices: ", len(indices))
    print(indices_list)
    for index in indices:
        img_path = image_paths[index]
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        #display image using matplotlib
        plt.imshow(img)
        plt.show()

def test_method(test_img_path):
    '''
    Test method that calls find_similar_images for a single image, given the image path of this image
    Wraps around find_similar_images(...) function and can be used for writing results to file
    '''
    test_img = Image.open(test_img_path).convert("RGB")
    # plt.imshow(test_img)
    # plt.show()
    indices_list,time_taken = find_similar_images(test_img_path, K_IMAGES, embedding, device)
    # write_to_file(test_img_path,indices_list,file_name)
    # testing_times.append(time_taken)
    return time_taken,indices_list

def test(image_paths):
    '''
    tests 100 random images and writes their results to a csv file
    '''
    testing_images=[]
    testing = pd.DataFrame({
            'testing_image': "dummy_data",
            'time_taken': 0,
            'indices_list': [],
            'time_taken': 0
        })
    for i in range(0,100):
        random_index=random.randint(0, len(image_paths)-1)
        testing_images.append(image_paths[random_index])

    for i in range(len(testing_images)):
        time_taken,indices_list=test_method(testing_images[i],args.resultFilePathJson)
        testing = testing.append(pd.DataFrame({
                    'testing_image': testing_images[i],
                    'time_taken': time_taken,
                    'output_indices' : list(indices_list),
                    'time_taken' : time_taken
                }), ignore_index=True)
    testing.to_csv(args.resultFilePath,index=False)

    return testing
        # print(time_taken)

def main():
    image_paths = load_images()
    testing_df = test(image_paths)

    return testing_df

if __name__ == "__main__":
    testing_df = main()