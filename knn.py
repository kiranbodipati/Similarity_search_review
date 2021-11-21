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

random.seed(10)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--imagesFilePath',default="./image-files/geological_mapping_data.json",type=str,help='image path files')
parser.add_argument('--resultFilePath',default="./knn-results/results.csv",type=str,help='file path to save results.')
parser.add_argument('--resultFilePathJson',default="./knn-results/results.json",type=str,help='file path to save results.')
parser.add_argument('--kNearest',default=10,type=int,help='top k images')
parser.add_argument('--encoderModelPath',default="./encoders/geological_encoding.pt",type=str,help='encoding model path')
parser.add_argument('--embeddingPath',default="./encoders/geological_embed.npy",type=str,help='embedding path')

args = parser.parse_args()

TEST_IMAGE_PATH = "/Users/abhishekvaidyanathan/Downloads/geological_similarity/schist/ZZ5Z5.jpg"
NUM_IMAGES = args.kNearest
ENCODER_MODEL_PATH = args.encoderModelPath
EMBEDDING_PATH = args.embeddingPath

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
encoder = ConvEncoder()
encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
encoder.eval()
encoder.to(device)
embedding = np.load(EMBEDDING_PATH)

def load_image_tensor(image_path, device):
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    #print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor

def load_images():
    image_paths=[]
    # with open("geological_map.json", 'r', encoding='utf-8') as f:
    #     image_paths=json.load(f)
    with open(args.imagesFilePath, 'r', encoding='utf-8') as f:
        image_paths=json.load(f)
    print(len(image_paths))
    return image_paths

def compute_similar_images(image_path, num_images, embedding, device):
    image_tensor = load_image_tensor(image_path, device)
    # image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    #print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    #print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="euclidean")
    knn.fit(embedding)
    start_time=time.time()
    _, indices = knn.kneighbors(flattened_embedding)
    end_time=time.time()
    time_taken=end_time-start_time
    #print("Time taken: ",end_time-start_time)
    indices_list = indices.tolist()
    
    #print(indices_list)
    return indices_list,time_taken

def plot_similar_images(image_paths,indices_list):
    indices = indices_list[0]
    print("total indices: ", len(indices))
    print(indices_list)
    for index in indices:
        # img_name = str(index - 1) + ".jpg"
        # print(img_name)
        img_path = image_paths[index]
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.show()

def test_method(test_img_path,file_name):
    test_img = Image.open(test_img_path).convert("RGB")
    # plt.imshow(test_img)
    # plt.show()
    indices_list,time_taken = compute_similar_images(test_img_path, NUM_IMAGES, embedding, device)
    # write_to_file(test_img_path,indices_list,file_name)
    #plot_similar_images(indices_list)
    # testing_times.append(time_taken)
    return time_taken,indices_list

def test(image_paths):
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