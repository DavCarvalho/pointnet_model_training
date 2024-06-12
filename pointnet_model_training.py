
#1 Setup Ambiente

#math library
import numpy as np

#drawing library
import matplotlib.pyplot as plt
import open3d as o3d

#Deep Learning Libraries
# torch é uma biblioteca de aprendizado de máquina de código aberto, usada para aplicações de aprendizado profundo como redes neurais.
import torch
# torch.nn.functional contém funções que não têm quaisquer parâmetros, como ativação ReLU, enquanto torch.nn contém classes que têm parâmetros que podem ser aprendidos, como camadas lineares e convolucionais.
import torch.nn.functional as nnf
import torch.nn as nn
import torchnet as tnt
# torch.optim é um pacote que implementa vários algoritmos de otimização, usados para treinar modelos de aprendizado de máquina.
import torch.optim as optim
# MultiStepLR é uma classe do PyTorch que ajusta a taxa de aprendizado em vários passos predefinidos.
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.neighbors import NearestNeighbors
# confusion_matrix é uma função que calcula a matriz de confusão para avaliar a precisão de uma classificação.
from sklearn.metrics import confusion_matrix

#utility libraries
# copy é um módulo Python que fornece funções para copiar objetos Python.
import copy
# glob é um módulo Python que encontra todos os caminhos que correspondem a um padrão especificado de acordo com as regras usadas pelo shell Unix.
from glob import glob
# os é um módulo Python que fornece uma maneira de usar funcionalidades dependentes do sistema operacional, como ler ou escrever em arquivos.
import os
# functools é um módulo para funções de ordem superior; funções que atuam ou retornam outras funções.
import functools
# mock é uma biblioteca para testar em Python. Permite que você substitua partes do seu sistema por objetos simulados e faça afirmações sobre como eles foram usados.
import mock
# tqdm é uma biblioteca Python que fornece uma barra de progresso visual para loops.
from tqdm.auto import tqdm
import time


#checar se cuda está disponível e set Pytorch para usar CPU ou GPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)


# 2 Preparando a Nuvem de Pontos
""" Aqui definimos
- load the previously prepared data
-Definimos a mesma função que foi usada para preparação dos dados, para carrergar os dados da nuvem de pontos.
-checamos se conseguimos acessar o labels de um elemento do train_set
"""
project_dir = "./DATA/"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.txt"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.txt"))

#function to load and normalize pointcloud data
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


#preparar os dados em conjuunto de treinamento, validation set (para tune parametros do modelo ) e o conjunto de teste (para evaluate o modelo)
# Escolhe aleatoriamente uma parte dos arquivos de treinamento para ser o conjunto de validação.
#O conjunto de validação é usado para verificar o quão bem nosso modelo está aprendendo durante o treinamento. Escolhemos 1/5 ou seja 20% dos arquivos
valid_index = np.random.choice(len(pointcloud_train_files), int(len(pointcloud_train_files) / 5 ), replace=False)
# Cria a lista de validação com os arquivos escolhidos
valid_list = [pointcloud_train_files[i] for i in valid_index]
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(list(range(len(pointcloud_train_files))), valid_index)]
test_list = pointcloud_test_files

print("%d tiles in train set, %d tiles in validation set, %d tiles in test set" % (len(train_list), len(valid_list), len(test_list)))

#gerar os conuunto de treino, teste e validação
cloud_features = "xyzrgbi"
test_set = tnt.dataset.ListDataset(test_list, functools.partial(cloud_loader, features_used=cloud_features))
train_set = tnt.dataset.ListDataset(train_list, functools.partial(cloud_loader, features_used=cloud_features))
valid_set = tnt.dataset.ListDataset(valid_list, functools.partial(cloud_loader, features_used=cloud_features))

#3- definindo a arquitetura da pointnet
def cloud_collate(batch):
    """
        Coleta  uma lista de amotras em uma lista de lote (batch) para nuvens 
        e um unico array para os labels
        ela é necessaria pois as nuvens tem diferentes tamanhos
    """
    clouds, labels = list(zip(*batch))
    labels = torch.cat(labels, 0)
    return clouds, labels

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
    

# 4- PointNet Class Arquiteture working test

#Cinfuguração parametros
# consideraremos a primeira nuvem do conjunto de treino
features_used = 'xyzrgbi'
cloud_data, gt = cloud_loader(train_list[10], features_used)

class_names = ['unclassified', 'vegetation', 'ground', 'buildings']

# para criar um input  apropiado para a pointnet nos precisamos adicionar uma dimensão vazia
#para o batch size (com palavra chave None) e subamostrar a nuvem de pontos para ter o tamanho subsample_size=512 pontos
#on top, precisa-se ter certeza que estamos trabalhando com double precision, (.float())
cloud_data = cloud_data[None,:,:512].float()

#criar o modelo com varios parametros
pnt = PointNet([32, 32], [32, 64, 256], [128, 64, 32], n_classes=len(class_names)-1, input_feat=len(features_used), subsample_size=512, cuda=1)
print(pnt)

#verificar se ta funcionandio corretamente com uma predição
pred = pnt(cloud_data)

#agora checamos se o tamanho esta de fato [n_batch, n_classes, subsample_size]
assert(pred.shape == torch.Size([1, len(class_names)-1, 512]))

# 5 Definição da Segmentação Semântica

#Definição do Classficador com PointNet

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
    

# 6- Code test 
"""
    Verificar se somos capazes de utlizar a classe PointCloudClassifier para classificar e diferrent point clouds
"""

# primeiro considerar a nuvem do conjunto de treino
features_used = 'xyzrgbi'
cloud1 = cloud_loader(train_list[0], features_used)
cloud2 = cloud_loader(train_list[1], features_used)

#artificialmente diminuir o tamanho da nuvem da cloud2 para testar ambos
#subsampling e oversampling cass
cloud2 = (cloud2[0][:,:200], cloud2[1][:200])
print(f"loading two clouds with {cloud1[0].shape[1]} and {cloud2[0].shape[1]} points")

batch_clouds, gt = cloud_collate([cloud1, cloud2])

#definir os argumentos para inicializar a instância
args = mock.Mock() # permiti preencher os diferentes argumentos
args.n_class = len(class_names)-1 # número de classes
args.input_feats = features_used # características usadas
args.subsample_size = 512 # número de pontos para subamostrar
args.cuda = 1 # se deve usar aceleração GPU

#criar a instanca do pointcloudclassifier
PCC = PointCloudClassifier(args)

#criar a instância do modelo
model = nn.Module() # inicializar o módulo com o vizinho mais próximo
model = PointNet([32, 32], [32, 64, 256], [128, 64, 32], n_classes=len(class_names)-1, input_feat=len(features_used), subsample_size=args.subsample_size, cuda=args.cuda)


#predição
pred = PCC.run(model, batch_clouds)

#Checamos se o tamanho da previsão esta de fato [sum_i, n_points_i, n_class]
assert(pred.shape == torch.Size([cloud1[0].shape[1] + cloud2[0].shape[1], len(class_names)-1]))

#7 Definição das métricas

#Definição da matriz de confusão
class ConfusionMatrix:
    def __init__(self, n_class, class_names):
        self.CM = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.class_names = class_names

    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add_batch(self, gt, pred):
        if len(gt) != len(pred):
            raise ValueError(f"Inconsistent number of samples: {len(gt)} in gt, {len(pred)} in pred")
        self.CM += confusion_matrix(gt, pred, labels=list(range(self.n_class)))

    def overall_accuracy(self):
        return 100 * self.CM.trace() / self.CM.sum()

    def class_IoU(self, show=1):
        ious = np.diag(self.CM) / (np.sum(self.CM, 1) + np.sum(self.CM, 0) - np.diag(self.CM))
        if show:
            print(' / '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
        return 100 * np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()



    
#8-Definição de funções de treino da pointet

def train(model, PCC, optimizer, args):
    """Train for one epoch"""
    model.train()

    # A função loader vai cuidar do batching
    loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # tqdm para a barra de progresso
    loader = tqdm(loader, ncols=100, desc='Training Epoch', leave=False)

    # Irá manter o curso da perda
    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names[1:])

    for index_batch, (cloud, gt) in enumerate(loader):
        if PCC.is_cuda:
            gt = gt.cuda()
        
        # Colocar o gradiente para zero
        optimizer.zero_grad()

        # Compute the prediction
        pred = PCC.run(model, cloud)

        labeled = gt != 0  # Remove os pontos unlabeled from supervision

        if labeled.sum() == 0:
            continue  # No labeled points, skip

        loss = nn.functional.cross_entropy(pred[labeled], gt[labeled] - 1)  # -1 to account for unlabeled class

        # Computar gradiente
        loss.backward()

        # Um passo de SGD
        optimizer.step()

        loss_meter.add(loss.item())

        # Precisa-se converter para numpy array, o que requer detaching gradientes e colocar de volta na RAM
        gt_labeled = gt[labeled].cpu().numpy() - 1  # -1 to account for unlabeled class
        pred_labeled = pred[labeled].argmax(1).cpu().detach().numpy()

        cm.add_batch(gt_labeled, pred_labeled)

    return cm, loss_meter.value()[0]


def eval(model, PCC, test, args):
    """
    Evaluate the model on the test set/ valid set
    """
    model.eval() # Coloca o modelo em modo de avaliação
    # Avaliação no conjunto de teste
    if test:
        loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=args.batch_size,
                                            shuffle=False)
        loader = tqdm(loader, ncols=500, leave=False, desc="Test")
    else:
        # Avaliação no conjunto de validação
        loader = torch.utils.data.DataLoader(valid_set, collate_fn=cloud_collate, batch_size=60, shuffle=False,
                                            drop_last=False)
        loader = tqdm(loader, ncols=500, leave=False, desc="Val")

    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names[1:])
    for index_batch, (cloud, gt) in enumerate(loader):
        # Como treino, sem gradientes
        if PCC.is_cuda:
            gt = gt.cuda()
        with torch.no_grad():
            pred = PCC.run(model, cloud)
        labeled = gt != 0  # Removemos os pontos não rotulados da supervisão
        if labeled.sum() == 0:
            continue  # Sem pontos rotulados, pular
        loss = nn.functional.cross_entropy(pred[labeled], gt[labeled]-1)
        loss_meter.add(loss.item())
        cm.add_batch((gt[labeled]-1).cpu(), pred[labeled].argmax(1).cpu().detach().numpy())

    return cm, loss_meter.value()[0]


def train_full(args):
    """The full training Loop"""
    # Inicializa o modelo
    model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args.n_class, input_feat=args.n_input_feats,
                     subsample_size=args.subsample_size)
    
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

    best_model = None
    best_mIoU = 0

    #defini o classificador
    PCC = PointCloudClassifier(args)

    #defni o otimizador
    #adam é um otimizador que ajusta a taxa de aprendizado de acordo com o gradiente
    #good guest para classificação
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #adding a scheduler for adaptive learning rate. at each milestone lr_setps

    TESTCOLOR = '\033[104m' #color for test
    TRAINCOLOR = '\033[100m' #color for train
    VALIDCOLOR = '\033[45m' #color for validation
    NORMALCOLOR = '\033[0m' #normal color

    metrics_pn = {}
    metrics_pn['definition'] =[['train_oa', 'train_mIoU', 'train_loss'], ['valid_oa', 'valid_mIoU', 'valid_loss'], ['test_oa', 'test_mIoU', 'test_loss']]

#nos garantimos aqui que sempre teremos a mlehor validação que peforma no nivel
    for i_epoch in tqdm(range(args.n_epoch), desc='Training'):
        # Treina um época
        cm_train, loss_train = train(model, PCC, optimizer, args)
        mIoU = cm_train.class_IoU()

        tqdm.write(TRAINCOLOR + 'Epoch %3d -> Train overwall accuracy : %3.2f%%, Train mIoU : %3.2f%%, Train Loss : %3.2f' % (i_epoch, cm_train.overall_accuracy(), mIoU, loss_train) + NORMALCOLOR)

        metrics_pn[i_epoch] = [[cm_train.overall_accuracy(), mIoU, loss_train]]

        #evaluate on the validation set
        cm_valid, loss_valid = eval(model, PCC, False, args=args)
        mIoU_valid = cm_valid.class_IoU()
        metrics_pn[i_epoch].append([cm_valid.overall_accuracy(), mIoU_valid, loss_valid])

        best_valid = False
        if mIoU_valid > best_mIoU:
            best_valid = True
            best_mIoU = mIoU_valid

            best_model = copy.deepcopy(model) #copy the stored model
            tqdm.write(VALIDCOLOR + 'Best perfomance achivied at epoch %3d -> Valid overwall accuracy : %3.2f%%, Valid mIoU : %3.2f%%, Valid Loss : %3.2f' % (i_epoch, cm_valid.overall_accuracy(), mIoU_valid, loss_valid) + NORMALCOLOR)
        else:
            tqdm.write(VALIDCOLOR + 'Epoch %3d -> Valid overwall accuracy : %3.2f%%, Valid mIoU : %3.2f%%, Valid Loss : %3.2f' % (i_epoch, cm_valid.overall_accuracy(), mIoU_valid, loss_valid) + NORMALCOLOR)
        
        
        if i_epoch == args.n_epoch-1 or best_valid:
            #evaluate on the test set
            cm_test, loss_test = eval(best_model, PCC, True, args=args)
            mIoU = cm_test.class_IoU()
            tqdm.write(TESTCOLOR + 'Epoch %3d -> Test overwall accuracy : %3.2f%%, Test mIoU : %3.2f%%, Test Loss : %3.2f' % (i_epoch, cm_test.overall_accuracy(), mIoU, loss_test) + NORMALCOLOR)
            
            metrics_pn[i_epoch].append([cm_test.overall_accuracy(), mIoU, loss_test])
    return best_model, metrics_pn


# 9 definindo os parametros de treinamento

#estrtura que vamos armazenar os parametros
args = mock.Mock()

#argumentos para experiment on
class_names = ['unclassified', 'vegetation', 'ground', 'buildings']
args.n_epoch = 50
args.subsample_size = 2048

#deixar os argumentos abaixo unchanged
args.n_epoch_test = int(1)
args.batch_size = 8
args.n_class = len(class_names)-1
args.input_feats = 'xyzrgbi'
args.n_input_feats = len(args.input_feats)
args.MLP_1 = [32, 32]
args.MLP_2 = [32, 64, 256]
args.MLP_3 = [128, 64, 32]
args.show_test = 0
args.lr = 5e-3
args.wd = 0
args.cuda = 1

# 10 PointNet Training
#60m para treinar com 30 epocas e 512 subsample size
t0 = time.time()
trained_model, metrics_pn = train_full(args)
t1 = time.time()

print(trained_model)
print(f"{'-'*50} ")
print(f"Total training time: {t1-t0} seconds")
print(f"{'='*50} ")



# 11- Pointnet Prediction Visualization
def tile_prediction(tile_name, model=None, PCC=None, Visualization=True, features_used='xyzrgbi'):
    # Load the tile
    cloud, gt = cloud_loader(tile_name, features_used)
    # Make the predictions
    labels = PCC.run(model, [cloud])
    labels = labels.argmax(1).cpu() + 1
    # Prepare the data for export
    xyz = np.array(cloud[0:3]).transpose()
    # Prepare the data for open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Visualization with Open3d
    if Visualization == True:
        max_label = labels.max()
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.estimate_normals(fast_normal_computation=True)
        o3d.visualization.draw_geometries([pcd])
    return pcd, labels

selection = test_list[9]
pcd, labels = tile_prediction(selection, model=trained_model, PCC=PCC)


#12 Exportar o modelo
# Salvar o peso do modelo

torch.save(trained_model.state_dict(), './pointnet_model_'+project_dir.split("DATA/")[1]+'.torch')

#salvando as métricas
with open("./metrics_"+project_dir.split("DATA/")[1]+".csv", "w") as f:
    for key, value in metrics_pn.items():
        f.write("%s,%s\n"%(key,value))