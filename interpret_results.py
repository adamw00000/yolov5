# %%
import os
import numpy as np
import pandas as pd

dfs = []
for filename in os.listdir('./inference/output'):
    if filename.split('.')[-1] != 'txt':
        continue
    df = pd.read_csv(f'./inference/output/{filename}', sep = ' ', header = None)
    df = df.drop(columns = [6])
    df.columns = ['Id', 'X', 'Y', 'Width', 'Height', 'Confidence']
    df = df.assign(Image = filename)
    dfs.append(df)

df = pd.concat(dfs).reset_index(drop = True)
df

# %%
df = df.loc[df \
        .groupby(['Image', 'Id'])\
        .Confidence.rank(axis = 1, 
            method = 'first', 
            ascending = False
        ) == 1
    ].sort_values(['Image', 'Id'])\
    .reset_index(drop = True)
df

# %%
test_path = 'data/valid.txt'

test_images = pd.read_csv(test_path, header = None)[0] \
    .str.replace(r'(.*\\)+', '') \
    .str.replace(r'\..*', '') \
    .values.tolist()
# test_images

# %%
import json
from PIL import Image
from pandas import json_normalize

image_dir = 'data/images'
label_dir = '../TXT'
def get_label(image_num):
    with open(os.path.join(label_dir, f'{image_num}.json')) as f:
        json_dict = json.load(f)
    
    df = json_normalize(json_dict['Landmarks'])
    df['IsValid'] = ~((df.X == 0) & (df.Y == 0))
    
    classes = ['S','N','A','B','O','cm1','cm2','As','Ai','OP6','Gn','Go','Pog','Po','Ar','PNS','ANS','Sn','Me','UL','LL','Pogp','Is','Ii','SOr','Si','Sp','Ba','Np','Te']
    # classes = ['S','N','B','O','Gn','Pog','Sn','Is','Si','Sp']

    df = df.loc[np.isin(df.Name, classes)].reset_index(drop = True)

    name_to_class_index = {v: k for k, v in dict(enumerate(classes)).items()}
    df.Id = df.Name.apply(lambda x: name_to_class_index[x])
    
    return df

def get_image(image_num):
    image_file = os.path.join(image_dir, f'{image_num}.jpg')
    im = Image.open(image_file)
    if im.mode != 'L':
        im = im.convert('L')
    
    return np.asarray(im)
    
X = {}
true_dfs = []
for image in dict(tuple(df.groupby('Image'))):
    image_num = image.split('.')[0]
    if not image_num in test_images:
        continue
    true_dfs.append(get_label(image_num).assign(Image = image))
    X[image_num] = get_image(image_num)

true_df = pd.concat(true_dfs)
true_df

# %%
pred_df = df.rename(columns = { 'X': 'XPred', 'Y': 'YPred' })
full_df = true_df.merge(pred_df, on = ['Image', 'Id'])
full_df

# %%
full_df_nan = true_df.merge(pred_df, on = ['Image', 'Id'], how = 'left')
full_df_nan.XPred.isna().mean()

# %%
y_true = full_df[['X', 'Y']].values
y_pred = full_df[['XPred', 'YPred']].values
threshold = 0.01

def near_distances(y_true, y_pred):
    distances = np.linalg.norm(y_pred - y_true, axis = 1)
    distances = distances[distances < 0.1]
    return np.mean(distances < threshold)

def mean_dist(y_true, y_pred):
    distances = np.linalg.norm(y_pred - y_true, axis = 1)
    return np.mean(distances)

near_distances(y_true, y_pred), mean_dist(y_true, y_pred)

# %%
import matplotlib.pyplot as plt

def plot_result(X, y_true, y_pred, index = 0, threshold = 0.01, 
        threshold_alpha = 0.15, marker_size = 1):
    
    plt.imshow(X, cmap='gray')
    h, w = X.shape

    # threshold circles
    points = y_true
    for row in points:
        circle = plt.Circle((row[0] * w, row[1] * h), threshold * h, 
            color='r', alpha = threshold_alpha)
        plt.gca().add_artist(circle)
    
    # markers
    points = y_true
    plt.scatter(points[:, 0] * w, points[:, 1] * h, s = marker_size, c = 'r')
    for row in points:
        plt.annotate(row[2], (row[0] * w, row[1] * h), c = 'r', fontsize = 3)
    points = y_pred
    plt.scatter(points[:, 0] * w, points[:, 1] * h, s = marker_size, c = 'y')
    for row in points:
        plt.annotate(row[2], (row[0] * w, row[1] * h), c = 'y', fontsize = 3)

for index in range(10):
    y_true_index = full_df[['X', 'Y', 'Name']].loc[full_df.Image == test_images[index] + '.txt'].values
    y_pred_index = full_df[['XPred', 'YPred', 'Name']].loc[full_df.Image == test_images[index] + '.txt'].values
    plot_result(X[test_images[index]], 
        y_true_index, 
        y_pred_index)
    # plt.show()
    plt.savefig(f'res_{index}.png', dpi = 450, bbox_inches = 'tight')
    plt.clf()

# %%
