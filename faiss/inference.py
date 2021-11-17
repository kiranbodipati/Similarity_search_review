import faiss # make faiss available
import numpy as np
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

image_paths=[]
with open("geological_map.json", 'r', encoding='utf-8') as f:
    image_paths=json.load(f)
print(len(image_paths))

embedding = np.load("geological_embed.npy")
random_img_index=random.randint(0, len(image_paths)-1)
xq=embedding[random_img_index]
xb=embedding
print(xb.shape)
print(xq.shape)
xq=np.reshape(xq, (1, 576))
print(xq.shape)
index = faiss.IndexFlatL2(576)# build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
k=10
Dist, Indices = index.search(xq, k)     # actual search
print(Indices)                   # neighbors of the 5 first queries


indices = Indices[0]
print("total indices: ", len(indices))
for index in indices:
    # img_name = str(index - 1) + ".jpg"
    # print(img_name)
    img_path = image_paths[index]
    print(img_path)
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    plt.show()