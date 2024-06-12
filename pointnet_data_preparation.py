import numpy as np
import random # vai ser para retirar do conjunto de treino algum tipo de conjunto de validação
import torch #blibioteca de deep learning pois é eficietne
import torchnet as tnt # type: ignore # para lidar com um monte de arquivos para colocar em uma variavel

#Plot Libraries
import open3d as o3d 

#Utility Libraries
from glob import glob
import os
import functools

# #Data Augmentation Libraries
# import cv2
# from skimage import io
# from skimage import transform as tf
# from skimage import img_as_ubyte
# from skimage.util import random_noise

#1-Data path setup

#especificar o data path e extrait o nome do arquivo
project_dir = "./DATA/"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.txt"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.txt"))

# # 2- Train, Test and Validation Set creation
# Seleciona aleatoriamente 20% dos índices dos arquivos de treinamento para serem usados como dados de validação.
valid_index = np.random.choice(len(pointcloud_train_files), int(len(pointcloud_train_files)/5), replace=False)

# Cria uma lista de arquivos de validação usando os índices selecionados.
valid_list = [pointcloud_train_files[i] for i in valid_index]

# Cria uma lista de arquivos de treinamento usando os índices que não foram selecionados para validação.
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(range(len(pointcloud_train_files)), valid_index)]

# Define a lista de arquivos de teste.
test_list = pointcloud_test_files

print("%d titles in train set- %d titles in valid set- %d  titles in valid test" % (len(train_list), len(valid_list), len(test_list)))

# Data Analysis
# Configura as opções de impressão do numpy para exibir 3 casas decimais
np.set_printoptions(precision=3)

# Seleciona aleatoriamente um arquivo da lista de arquivos de treinamento de nuvem de pontos
tile_selected = pointcloud_train_files[random.randrange(20)]
print("Title selected: ", tile_selected)  # Imprime o nome do arquivo selecionado

# Carrega os dados do arquivo selecionado em um array numpy
temp = np.loadtxt(tile_selected)

# Imprime a mediana de cada coluna dos dados
print("median\n", np.median(temp, axis=0))

# Imprime o desvio padrão de cada coluna dos dados
print("std\n", np.std(temp, axis=0))

# Imprime o valor mínimo de cada coluna dos dados
print("min\n", np.min(temp, axis=0))

# Imprime o valor máximo de cada coluna dos dados
print("max\n", np.max(temp, axis=0))


# Computing the mean and the min of a data title
# Transpõe os dados para facilitar a seleção de índices. 
# Isso significa que agora cada linha representa uma característica (por exemplo, x, y, z, intensidade) e cada coluna representa um ponto na nuvem de pontos.
cloud_data = temp.transpose()

# Calcula o valor mínimo de cada característica.
min_f = np.min(cloud_data, axis=1)
# Calcula a média de cada característica.
mean_f = np.mean(cloud_data, axis=1)

# Imprime o valor mínimo de cada característica.
print("min transpose\n", min_f)
# Imprime a média de cada característica.
print("mean transpose\n", mean_f)

# Normaliza as coordenadas da nuvem de pontos.
# Primeiro, seleciona as três primeiras linhas dos dados, que representam as coordenadas x, y e z.
n_coords = cloud_data[0:3]

# Subtrai a média das coordenadas x e y, e o valor mínimo da coordenada z.
# Isso tem o efeito de centralizar a nuvem de pontos em torno da origem (0,0,0).
n_coords[0] -= mean_f[0]
n_coords[1] -= mean_f[1]
n_coords[2] -= min_f[2] # "nivelar" a nuvem de pontos ao longo do eixo z, garantir que todas as nuvens de pontos estejam na mesma escala e posição relativa.

# Imprime as coordenadas normalizadas.
print("n_coords\n", n_coords)

# Normalizar a intensidade
# para ter algo mais robusto e escalável
# o que signica que usaremos  a diferença antártil e essa é a diferença entre 75 e o 25 quantile

# A diferença interquartil (IQR) é a diferença entre o 75º e o 25º percentil.
# Estamos calculando isso para a intensidade dos pontos na nuvem de pontos.
# cloud_data[-2] se refere à penúltima coluna dos dados, que presumivelmente contém os valores de intensidade.
IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)

# Subtraímos a mediana de todas as observações e depois dividimos pela diferença interquartil.
# Isso tem o efeito de escalar os dados de forma que a mediana seja 0 e a maioria dos dados esteja entre -0.5 e 0.5.
# Isso também tem o efeito de reduzir o impacto de outliers (valores que são significativamente maiores ou menores que a maioria dos outros valores).
n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)

# Subtraímos o valor mínimo da intensidade normalizada.
# Isso garante que todos os valores sejam positivos, com o menor valor sendo 0.
n_intensity -= np.min(n_intensity)

# Imprime a intensidade normalizada.
print("n_intensity\n", n_intensity)

# Cloud load function
#para cada nuvem que esta carregada queremos aplicar a normalização

# A função cloud_loader recebe o nome do arquivo (tile_name) e uma lista de características a serem usadas (features_used).
def cloud_loader(tile_name, features_used):
    # Carrega os dados do arquivo e transpõe o array resultante.
    # A transposição é feita para facilitar a manipulação dos dados mais tarde.
    cloud_data = np.loadtxt(tile_name).transpose()

    # Calcula o valor mínimo e a média de cada característica.
    min_f = np.min(cloud_data, axis=1)
    mean_f = np.mean(cloud_data, axis=1)

    # Cria uma lista vazia para armazenar as características selecionadas.
    features = []

    # Se 'xyz' estiver na lista de características usadas, normaliza as coordenadas x, y e z e as adiciona à lista de características.
    if 'xyz' in features_used:
        n_coords = cloud_data[0:3]
        n_coords[0] -= mean_f[0]
        n_coords[1] -= mean_f[1]
        n_coords[2] -= min_f[2]
        features.append(n_coords)

    # Se 'rgb' estiver na lista de características usadas, adiciona as cores à lista de características.
    if 'rgb' in features_used:
        colors = cloud_data[3:6]
        features.append(colors)

    # Se 'i' estiver na lista de características usadas, normaliza a intensidade e a adiciona à lista de características.
    if 'i' in features_used:
        IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)
        n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
        n_intensity -= np.min(n_intensity)
        features.append(n_intensity)

    # A última linha dos dados é assumida como a verdade fundamental (ground truth) e é convertida para um tensor PyTorch.
    gt = cloud_data[-1]
    gt = torch.from_numpy(gt).long()

    # As características selecionadas são empilhadas verticalmente e convertidas para um tensor PyTorch.
    cloud_data = torch.from_numpy(np.vstack(features))

    # A função retorna a nuvem de pontos pré-processada e a verdade fundamental.
    return cloud_data, gt


print("cloud_loader function created", cloud_loader(tile_selected, "xyzrgbi"))

# train, test, validation split dataset
# Define as características que serão usadas para carregar os dados da nuvem de pontos.
cloud_features = "xyzrgbi"

# Cria um conjunto de dados de teste. 
# A função cloud_loader é usada para carregar os dados, com as características definidas acima.
# A lista test_list contém os nomes dos arquivos que serão usados para o conjunto de teste.
test_set = tnt.dataset.ListDataset(test_list, functools.partial(cloud_loader, features_used=cloud_features))

# Cria um conjunto de dados de treinamento de maneira semelhante ao conjunto de teste.
train_set = tnt.dataset.ListDataset(train_list, functools.partial(cloud_loader, features_used=cloud_features))

# Cria um conjunto de dados de validação de maneira semelhante aos conjuntos de teste e treinamento.
valid_set = tnt.dataset.ListDataset(valid_list, functools.partial(cloud_loader, features_used=cloud_features))

# Imprime o primeiro elemento do conjunto de teste.
print("test_set", test_set[0])


# Title visualization function
# Define a função de visualização do título.
def title_visualization(title_name, features_used='xyzrgbi'):
    # Carrega a nuvem de pontos e os dados de ground truth (gt) usando a função cloud_loader.
    cloud, gt = cloud_loader(title_name, features_used)

    # Prepara os dados para o Open3D, uma biblioteca usada para visualização 3D.
    # Extrai as coordenadas x, y, z da nuvem de pontos e as transpõe para o formato correto.
    xyz = np.array(cloud[0:3]).transpose()
    # Cria um objeto PointCloud, que é usado pelo Open3D para armazenar e manipular nuvens de pontos.
    pcd = o3d.geometry.PointCloud()
    # Atribui as coordenadas x, y, z à nuvem de pontos.
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Se as cores RGB estão presentes nos dados, extrai-as e as normaliza para o intervalo [0, 1].
    if 'rgb' in features_used:
        rgb = np.array(cloud[3:6]/255).transpose()
        # Atribui as cores à nuvem de pontos.
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Estima as normais da nuvem de pontos, que são usadas para melhorar a visualização 3D.
    pcd.estimate_normals(fast_normal_computation=True)
    # Desenha a nuvem de pontos usando o Open3D.
    o3d.visualization.draw_geometries([pcd])

    return

# Seleciona um título da lista de testes.
selection = test_list[9]
# Chama a função de visualização do título com o título selecionado.
title_visualization(selection)