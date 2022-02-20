import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import requests
import click

from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial import KDTree, SphericalVoronoi


class Target:
    """
    Simple class, store name, img and embedding
    """
    name = ''
    img = []
    emb = []

def init_img(fpath, mtcnn, resnet):
    """
    Crops and scale image, calculate embedding.
    Returns embedding and cropped img.
    """
    imgdir='./Data'
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    # Image loading and preprocessing
    img = Image.open(fpath)
    img_cropped = mtcnn(img)

    # Calc embedding
    img_embedding = resnet(img_cropped.unsqueeze(0))
    img_embedding = img_embedding.squeeze().detach().numpy()
    # Image rescale
    img_cropped = img_cropped.permute(1,2,0).numpy()
    img_cropped = (img_cropped + 1) / 2

    box = np.round(mtcnn.detect(img)[0][0]).astype(np.int32)
    box_x = [box[0], box[2], box[2], box[0], box[0]]
    box_y = [box[1], box[1], box[3], box[3], box[1]]
    x0, x1 = box[0], box[2]
    y0, y1 = box[1], box[3]

    img_cropped_rescale = np.array(img)[y0:y1, x0:x1]
    return img_embedding, img_cropped_rescale


def find_nn(dump, trg, n=10):
    """Finds Nearest Neighbot"""
    # KDtree initialization
    emb = np.vstack(dump.embedding)
    count = 0
    r = 0.9
    tree = KDTree(emb)
    while count < n:
        nearest = tree.query_ball_point(trg.emb, r=r, p=2)    # Euler metric
        r += 0.1
        count = len(nearest)

    idxs, dists = [], []
    for i in nearest:
        [idx, n_emb] = list(dump.iloc[i][['id', 'embedding']])
        dists.append(get_distance(trg.emb, n_emb))
        idxs.append(int(idx))

    df_img = pd.DataFrame({
        'id': idxs,
        'dist': dists
    }).sort_values('dist')

    n_col, n_row = find_optimal_grid(n+1, 2)
    fsize = 5
    plt.figure(dpi=200)
    plt.subplot(n_col, n_row, 1)
    plt.title('Target',fontdict = {'fontsize' : fsize}, pad=0)
    plt.imshow(trg.img)
    plt.axis('off')

    for i in range(min(n, len(df_img))):

        [idx, dist] = list(df_img.iloc[i][['id', 'dist']])
        idx = int(idx)
        img = download_by_idx(idx)

        plt.subplot(n_col, n_row, i+2)
        plt.title(f'ID{idx}',fontdict = {'fontsize' : fsize}, pad=2)
        plt.imshow(img)
        plt.axis('off')

    res_dir = './output'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    fname = f'{res_dir}/result_by_{trg.name.split(".")[0]}.png'
    plt.savefig(fname)
    plt.show()


def download_by_idx(idx):
    """Download image by idx from kino-teatr.ru"""
    filename = f'{idx}.jpg'
    url_portal = 'http://www.kino-teatr.ru/acter/foto/ros/'
    url_img = f'{url_portal}{filename}'
    img_data = requests.get(url_img).content
    fpath = f'data/{filename}'
    with open(fpath, 'wb') as handler:
        handler.write(img_data)
    
    return np.array(Image.open(fpath))

def find_optimal_grid(n=10, aspect=2):
    """Generate subplot grid for result image"""
    c = 1
    r = np.round(c * aspect)
    n_grid = c * r
    while n_grid < n:
        c += 1
        r = np.round(c * aspect)
        n_grid = c * r
    return c, r


def get_distance(a, b):
    """Calculate rms metric"""
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum(np.power(a - b, 2)))

def_dump = 'russian_actors_facenet_embeddings_27085.pkl'
def_img = 'pitt_2.jfif'
@click.command()
@click.option('-dump', default=def_dump, help='Name embedding dump.')
@click.option('-img', default=def_img, help='Path to image.')
@click.option('-n', default=30, help='Required number of actor.')
def main(dump, img, n = 10):
    """
    Finds N target neighbors by embeding dump
    """
    # Initialize embedings and img
    dump = pd.read_pickle(dump)
    
    mtcnn = MTCNN(image_size=150, margin=10)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    trg = Target()
    trg.name = img
    trg.emb, trg.img = init_img(trg.name, mtcnn, resnet)
    find_nn(dump, trg, n)

if __name__ == '__main__':
    main()