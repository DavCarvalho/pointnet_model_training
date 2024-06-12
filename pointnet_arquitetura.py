#Blibioteca de Matemática Básica
import numpy as np

# Blibioteca de Deep Learning
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors #that be used simply at one point, we will sub sample the titles to a fixed number
                                            # we habe to assure that we can perform the nearest neighbor search in a fast way
# We want to label each point in the original Point Cloud which means that we need to infer the prediction  and to do that for all points
# that are not presented in a subsampled case that is labeled and fot that god you can always upscale and thats at the hyperparameter function at the end

def cloud_loader(tile_name, features_used):
    cloud_data = np.loadtxt(tile_name).transpose()

    min_f = np.min(cloud_data, axis=1)
    mean_f = np.mean(cloud_data, axis=1)

    features = []
    if 'xyz' in features_used:
        n_coords = cloud_data[0:3]
        n_coords[0] -= mean_f[0]
        n_coords[1] -= mean_f[1]
        n_coords[2] -= min_f[2]
        features.append(n_coords)
    if 'rgb' in features_used:
        colors = cloud_data[3:6]
        features.append(colors)
    if 'i' in features_used:
        IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)
        n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
        n_intensity -= np.min(n_intensity)
        features.append(n_intensity)
    gt = cloud_data[-1]
    gt = torch.from_numpy(gt).long()
    cloud_data = torch.from_numpy(np.vstack(features))
    return cloud_data, gt


class PointNet(nn.Module):
    # Pointnet for semantic segmentation
    def __init__(self, MLP_1, MLP_2, MLP_3, n_classes=3, input_feat=3, subsample_size=512, cuda=1):
        """
        initialization function
        MLP_1, MLP 2 and MLP_3: int list = width of the Layers of
        multi-layer perceptrons. For example MLP_1 = [32, 64] or [16, 64, 128] n_class: int= the number of class
        input_feat: int = number of input feature
        subsample size: int number of points to which the tiles are subsampled 
        cuda: int = if e, run on CPU (slow but easy to debug), if 1 on GPU
        """
        super(PointNet, self).__init__() # necessario para todas as classes que extendem o modulo class

        # cuda is super important because it paralyze a lot of computation 
        # we need that for training but you dont need to do that for inference
        # so we havbe to check if we want to use cuda or no
        self.is_cuda = cuda
        self.subsample_size = subsample_size

        # como não conhecemos o número de camadas, vamos usar um loop para criar as camadas
        # para criar o número correto de camadas, isso signfica o loop sobre o comprimento do MLP e para cada layer vamso criar uma camada de convolução 1D
        # totalmente conectada

        # bassicaly we embed the initial feature vector into a new feature space with a new feature vector
        # and thats when we take the last element of the list
        m1 = MLP_1[-1] # tamanho do primeiro elemento embeding F1
        m2 = MLP_2[-1] # tamanho do segundo elemento embeding F2

        # MLP_1 input [input_feature x n] -> f1 [m1 x n] -> [3 x n] = (64, 64)
        modules = []
        for i in range(len(MLP_1)): # loop sobre as camadas MLP1
            # note: para a primeira camada, o first in_channels is feature_size
            # Camada de convolução 1D
            modules.append(
                nn.Conv1d(
                    in_channels=MLP_1[i-1] if i>0 else input_feat,# truque para a single line
                    out_channels=MLP_1[i],
                    kernel_size=1
                )
            )
            modules.append(nn.BatchNorm1d(MLP_1[i]))
            modules.append(nn.ReLU())
        # transforma a lista de camadas em um callable modulo
        self.mlp_1 = nn.Sequential(*modules)

        # MLP_2 f1 [m1 x n] -> f2 [m2 x n] = (64, 128, 1024)
        modules = []
        for i in range(len(MLP_2)):
            modules.append(nn.Conv1d(MLP_2[i-1] if i>0 else m1, MLP_2[i], 1))
            modules.append(nn.BatchNorm1d(MLP_2[i]))
            modules.append(nn.ReLU(True))
            
        self.mlp_2 = nn.Sequential(*modules)

        # MLP_3 f1: [(m1 + m2) * n] -> output [k x n]
        modules = []
        for i in range(len(MLP_3)):
            modules.append(nn.Conv1d(MLP_3[i-1] if i>0 else m1+m2, MLP_3[i], 1))
            modules.append(nn.BatchNorm1d(MLP_3[i]))
            modules.append(nn.ReLU(True))
        # nota: a ultima camada não tem normalização or ativação
        modules.append(nn.Conv1d(MLP_3[-1], n_classes, 1))
        self.mlp_3 = nn.Sequential(*modules)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

        if cuda:
            self = self.cuda()

    def forward(self, input):
        """
        the foward function producing the embeddings for each point of 'input'
        input: [n_batch, input_feat, subsample_size] float array  = input_features
        output: [n_batch, n_classes, subsample_size] float array = point class logits
        """

        # print(f'input_size: {input}')
        if self.is_cuda: # put the input on the GPU
            input = input.cuda()

        # embed points , equation (1)
        # print("feature 1")
        f1 = self.mlp_1(input)

        # print("feature 2")
        # second points embedding, equation (2)
        f2 = self.mlp_2(f1)

        # print('maxpooling')
        # maxpooling, equation (3)
        G = self.maxpool(f2)

        # print('concatenation')
        # concatenate f1 and G
        #before
        #Gf1 = torch.cat((G.repeat(1,1,self.subsample_size), f1),1)
        G = G.repeat(1, 1, f1.size(2))
        Gf1 = torch.cat((G, f1), 1)

        # print("global + local featuring")
        # equation (4)
        out = self.mlp_3(Gf1)
        return out

class PointCloudClassifier:
    """"A principal classe de classificador de nuvem de pontos
    lida com a subamostragem dos títulos para um número fixo de pontos
    e interpolação para a nuvem de pontos original"""

    def __init__(self, args):
        self.subsample_size = args.subsample_size # número de pontos para subamostrar cada nuvem de pontos nos lotes
        self.n_inputs_feats = 3 # tamanho dos descritores de pontos na entrada
        if 'i' in args.input_feats: # adiciona intensidade
            self.n_inputs_feats += 1
        if 'rgb' in args.input_feats: # adiciona cor
            self.n_inputs_feats += 3
        self.n_class = args.n_class # número de classes na previsão
        self.is_cuda = args.cuda # se deve usar aceleração GPU

    def run(self, model, clouds):
        """
        ENTRADA:
        model = a rede neural
        clouds = lista de tensores n_batch de tamanho [n_feat, n_points_i]: lote de nuvens de pontos n_batch de tamanho n_points_i
        SAÍDA:
        pred = tensor float [sum_i n_points_i, n_class]: previsão para cada elemento do lote concatenado em um único tensor
        """

        # número de lotes
        n_batch = len(clouds)

        # conterá a previsão para todas as nuvens no lote (a saída)
        prediction_batch = torch.zeros((self.n_class, 0))

        # sampled_cloud contém as nuvens do lote, cada uma subamostrada para self.subsample_size pontos
        sampled_clouds = torch.Tensor(n_batch, self.n_inputs_feats, self.subsample_size)

        if self.is_cuda:
            prediction_batch = prediction_batch.cuda()

        for i_batch in range(n_batch):
            # carrega os elementos no lote um por um e subamostra/sobreamostra eles
            # para um tamanho de self.subsample_size pontos

            cloud = clouds[i_batch][:,:] # todos os pontos na nuvem

            n_points = cloud.shape[1] # número de pontos na nuvem

            selected_points = np.random.choice(n_points, self.subsample_size) # seleciona os pontos para manter
            sampled_cloud =  cloud[:, selected_points] # reduz a nuvem atual para pontos selecionados

            sampled_clouds[i_batch,:,:] = sampled_cloud # adiciona a nuvem de amostra a sampled_clouds

        # agora temos um tensor contendo um lote de nuvens com o mesmo tamanho
        sampled_prediction = model(sampled_clouds) # classifica o lote de nuvens amostradas

        # interpolação para as nuvens de pontos originais
        for i_batch in range(n_batch):
            # a nuvem de pontos original (apenas posição xyz)
            cloud = clouds[i_batch][:3,:]
            # e o lote amostrado correspondente (apenas posição xyz)
            sampled_cloud = sampled_clouds[i_batch,:3,:]

            # agora interpola a previsão de pontos de 'sampled_cloud' para a nuvem de pontos original "cloud"
            # com interpolação knn
            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sampled_cloud.cpu().permute(1,0))
            # seleciona para cada ponto na nuvem de pontos original o ponto mais próximo na nuvem amostrada
            dump, closest_point = knn.kneighbors(cloud.permute(1,0).cpu())
            # remove dimensão desnecessária
            closest_point = closest_point.squeeze()

            # previsão para a nuvem de pontos original i_batch ->
            # cada ponto na nuvem de pontos original recebe o rótulo do ponto mais próximo
            # na nuvem amostrada
            prediction_full_cloud = sampled_prediction[i_batch,:, closest_point]

            # anexa previsão full_cloud a prediction_batch
            prediction_batch = torch.cat((prediction_batch, prediction_full_cloud), 1)

        return prediction_batch.permute(1,0)










                                    




   