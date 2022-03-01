import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import requests
import click

from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial import KDTree, SphericalVoronoi

plt.ion()

class Target:
    """
    Simple class, store name, img and embedding
    """
    name = ''
    img = []
    emb = []

def get_output_path(fname, postfix = ''):
    res_dir = './output'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    name = fname.replace('\\', '/').split('/')[-1].split(".")[0]
    fname_out = f'{res_dir}/{name}_{postfix}.jpg'
    return fname_out

def init_img(fpath, mtcnn, resnet):
    """
    Crops and scale image, calculate embedding.
    Returns embedding and cropped img.
    """
    imgdir='./data'
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    # Image loading and preprocessing
    img = Image.open(fpath)


    img_cropped = mtcnn(img)

    if img_cropped == None:
        plt.figure()
        plt.imshow(img)
        plt.xticks([]);  plt.yticks([])
        fname_out = get_output_path(fpath, 'detect')
        plt.savefig(fname_out)
        plt.show(block=True)

        print('No face found')

        status = 0
        img_embedding, img_cropped_rescale = [], []

    else:

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


        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.xticks([]);  plt.yticks([])

        plt.subplot(1,3,2)
        plt.imshow(img)
        plt.plot(box_x, box_y, 'r', lw=2)
        plt.xticks([]);  plt.yticks([])

        plt.subplot(1,3,3)
        plt.imshow(img_cropped_rescale)
        plt.xticks([]);  plt.yticks([])
        fname_out = get_output_path(fpath, 'detect')
        plt.savefig(fname_out)
        plt.show()

        status = 1

    return img_embedding, img_cropped_rescale, status


def find_nn(dump, trg, n=10):
    """Finds Nearest Neighbours by KDtree"""
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
        plt.title(f'ID {idx}',fontdict = {'fontsize' : fsize}, pad=2)
        plt.imshow(img)
        plt.axis('off')

    fname = get_output_path(trg.name, 'output')
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

def download_from_gdrive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_content(response, destination)

def get_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def_token = '1Ho4nnoetBUvSff98H4lmO7tRAo95EK7m'
def_img = 'input/16454637998023702618065703043527.jpg'
# def_img = 'input/pitt_2.jfif'

@click.command()
@click.option('-dump', default=def_token, help='Name embedding dump.')
@click.option('-img', default=def_img, help='Path to image.')
@click.option('-n', default=30, help='Required number of actor.')
def main(dump, img, n = 10):
    """
    Finds N target neighbors by embeding dump
    """
    # Initialize embedings and img
    if not'.pkl' in dump:   # Google drive token as arg
        def_dump = 'russian_actors_facenet_embeddings.pkl'
        if not os.path.exists(def_dump):
            download_from_gdrive(dump,def_dump)
        dump = def_dump
    dump = pd.read_pickle(dump)

    mtcnn = MTCNN(image_size=150, margin=10)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    trg = Target()
    trg.name = img
    trg.emb, trg.img, status = init_img(trg.name, mtcnn, resnet)
    if status:
        find_nn(dump, trg, n)

if __name__ == '__main__':
    main()