import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.neighbors as NearestNeighbors

class Tnet(nn.Module):
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()
        self.dim = dim 
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.max_pool(x).view(bs, -1)
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x

class PointNet(nn.Module):
    def __init__(self, MLP_1, MLP_2, MLP_3, n_classes=3, input_feat=3, subsample_size = 51, cuda=1):
        super(PointNet, self).__init__()
        self.is_cuda = cuda
        self.subsample_size = subsample_size
        m1 = MLP_1[-1]
        m2 = MLP_2[-1]
        modules = []
        for i in range(len(MLP_1)):
            modules.append(nn.Conv1d(MLP_1[i-1] if i>0 else input_feat, MLP_1[i], 1))
            modules.append(nn.BatchNorm1d(MLP_1[i]))
            modules.append(nn.ReLU())
        self.mlp_1 = nn.Sequential(*modules)
        modules = []
        for i in range(len(MLP_2)):
            modules.append(nn.Conv1d(MLP_2[i-1] if i>0 else m1, MLP_2[i], 1))
            modules.append(nn.BatchNorm1d(MLP_2[i]))
            modules.append(nn.ReLU(True))
        self.mlp_2 = nn.Sequential(*modules)
        modules = []
        for i in range(len(MLP_3)):
            modules.append(nn.Conv1d(MLP_3[i-1] if i>0 else m1+m2, MLP_3[i], 1))
            modules.append(nn.BatchNorm1d(MLP_3[i]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Conv1d(MLP_3[-1], n_classes, 1))
        self.mlp_3 = nn.Sequential(*modules)
        self.maxpool = nn.MaxPool1d(subsample_size)
        self.tnet1 = Tnet(dim=3, num_points=subsample_size)
        self.tnet2 = Tnet(dim=64, num_points=subsample_size)
        if cuda:
            self = self.cuda()

    def forward(self, input):
        if self.is_cuda:
            input = input.cuda()
        bs = input.shape[0]
        A_input = self.tnet1(input)
        input = torch.bmm(input.transpose(2, 1), A_input).transpose(2, 1)
        f1 = self.mlp_1(input)
        A_feat = self.tnet2(f1)
        f1 = torch.bmm(f1.transpose(2, 1), A_feat).transpose(2, 1)
        f2 = self.mlp_2(f1)
        G = self.maxpool(f2)
        Gf1 = torch.cat((G.repeat(1,1,self.subsample_size), f1),1)
        out = self.mlp_3(Gf1)
        return out
            

class PointCloudClassifier:
        """"
        the main point of classifier class
        deal with subsampling the titles to a fixed number of points
        and interpolation to the original point cloud
        """

        def __init__(self, args):
                self.subsample_size = args.subsample_size # number of points to subsample each point cloud in the batches
                self.n_inputs_feats = 3 #size of points descriptors in input
                if 'i' in args.input_feats: #add itensity
                        self.n_inputs_feats += 1
                if 'rgb' in args.input_feats: #add color
                        self.n_inputs_feats += 3
                self.n_class = args.n_class #number of classes in the prediction
                self.is_cuda = args.cuda # wether to use GPU acceleration

        def run(self, model, clouds):
                """
                INPUT:
                model = the neural network
                clouds = list of n_batch tensors of size [n_feat, n_points_i]: batch of n_batch point clouds of size n_points_i
                OUTPUT:
                pred = [sum_i n_points_i, n_class] float tensor: prediction for each element of the batch concatenated in a single tensor
                """

                #number of batches
                n_batch = len(clouds)

                # will contain the prediciton for all clouds in batch (the output)
                prediction_batch = torch.zeros((self.n_class, 0))

                #sampled_cloud contains the clouds from the batch, each subsampled to self.subsample_size points
                sampled_clouds = torch.Tensor(n_batch, self.n_inputs_feats, self.subsample_size)

                if self.is_cuda:
                        prediction_batch = prediction_batch.cuda()
                
                for i_batch in range(n_batch):
                        #laod the elements in the batch one by one and subsample/oversmaple them
                        # to a size of self.subsample_size points

                        cloud = clouds[i_batch][:,:] # all points in the cloud

                        n_points = cloud.shape[1] # number of points in the cloud

                        selected_points = np.random.choice(n_points, self.subsample_size) #select the points to keep
                        sampled_cloud =  cloud[:, selected_points] #reduce the current cloud to selected points

                        sampled_clouds[i_batch,:,:] = sampled_cloud # add the sample cloud to sampled_clouds
            
                # we now have a tensor containing a batch of clouds with the same size
                sampled_prediction = model(sampled_clouds) #classify the vatch of  sampled clouds

                #interpolation to the original point clouds
                for i_batch in range(n_batch):
                        #the original point cloud (only xyz position)
                        cloud = clouds[i_batch][:3,:]
                        #and the corresponding sampled batch (only xyz position)
                        sampled_cloud = sampled_clouds[i_batch,:3,:]

                        # now interpolate the prediction of points of 'sampled_cloud' to the original point cloud "cloud"
                        # with knn interpolation
                        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sampled_cloud.cpu().permute(1,0))
                        #select for each point in the original point cloud the closest point in the sampled cloud
                        dump, closest_point = knn.kneighbors(cloud.permute(1,0).cpu())
                        #remove uneeeded dimension
                        closest_point = closest_point.squeeze()

                        #prediciton for the original point cloud i_batch ->
                        # each point in the original point cloud get the label of the closest point
                        # in the sampled cloud
                        prediction_full_cloud = sampled_prediction[i_batch,:, closest_point]

                        #append prediction full cloud to  prediciton batch

                        prediction_batch = torch.cat((prediction_batch, prediction_full_cloud), 1)

                return prediction_batch.permute(1,0)










                                    




   