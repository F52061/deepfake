import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch,warnings,argparse
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from vit_adaptive_mattn_aps import vit_base_patch16_224,vit_base_patch16_224_in21k
from dataset import MyDataset,albu_transforms,TrainDataset,TestDataset
import random
import numpy as np
warnings.filterwarnings('ignore')
torch.set_num_threads(2)
num_classes = 2
batch_size = 64
EPOCH = 50
pre_epoch = 0
input_size = 224

class MainModel(nn.Module):
    def __init__(self,pretrained=True):
        super(MainModel,self).__init__()
        self.model = vit_base_patch16_224(pretrained=pretrained,num_classes=2)
        # for name, param in self.model.named_parameters():
        #     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

    def forward(self, x,return_attn=False,aps=None,only_feat=False):
        if only_feat:
            _,_,x = self.model.forward(x,return_attn=True,aps=aps)
            return x
        x = self.model.forward(x,return_attn,aps)
        return x

parser = argparse.ArgumentParser(description='PyTorch DeepNetwork Training')
parser.add_argument('--outf', default='./results/Ama1_aps1_1', help='folder to output images and model checkpoints')  # 
args = parser.parse_args([])
if not os.path.exists(args.outf):
    os.makedirs(args.outf)

def main(only_cd=True):
    LR = 4e-6  #3e-5, 4e-6
    train_batch = 60
    print("Start Training, DeepNetwork!")  
    training_set_df = TrainDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/ffpp_train_split_8.txt',transforms=albu_transforms(),expected_method='Deepfakes')#Face2Face,Deepfakes,FaceSwap,NeuralTextures,original_sequences
    training_generator_df = torch.utils.data.DataLoader(training_set_df,batch_size=train_batch//5,shuffle=True)
    training_set_f2f = TrainDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/ffpp_train_split_8.txt',transforms=albu_transforms(),expected_method='Face2Face')
    training_generator_f2f = torch.utils.data.DataLoader(training_set_f2f,batch_size=train_batch//5,shuffle=True)
    training_set_fs = TrainDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/ffpp_train_split_8.txt',transforms=albu_transforms(),expected_method='FaceSwap')
    training_generator_fs = torch.utils.data.DataLoader(training_set_fs,batch_size=train_batch//5,shuffle=True)
    training_set_nt = TrainDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/ffpp_train_split_8.txt',transforms=albu_transforms(),expected_method='NeuralTextures')
    training_generator_nt = torch.utils.data.DataLoader(training_set_nt,batch_size=train_batch//5,shuffle=True)
    training_set_or = TrainDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/ffpp_train_split_8.txt',transforms=albu_transforms(),expected_method='original_sequences')
    training_generator_or = torch.utils.data.DataLoader(training_set_or,batch_size=train_batch//5,shuffle=True)

    val_set_ffpp = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/ffpp_test_split.txt')
    val_loader_ffpp = torch.utils.data.DataLoader(val_set_ffpp, batch_size=batch_size, shuffle=True)
    val_set_cd2 = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/CD2_test.txt')
    val_loader_cd2 = torch.utils.data.DataLoader(val_set_cd2, batch_size=batch_size, shuffle=True)
    val_set_dp = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/dfdcp_test.txt')
    val_loader_dp = torch.utils.data.DataLoader(val_set_dp, batch_size=batch_size, shuffle=True)
    val_set_wdf = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/wild_test.txt')
    val_loader_wdf = torch.utils.data.DataLoader(val_set_wdf, batch_size=batch_size, shuffle=True)
    val_set_ffiw = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/FFIW_test.txt')
    val_loader_ffiw = torch.utils.data.DataLoader(val_set_ffiw, batch_size=batch_size, shuffle=True)
    
    if only_cd:
        val_loaders = [val_loader_ffpp,val_loader_cd2,val_loader_dp,val_loader_wdf,val_loader_ffiw]
        val_names = ['ffpp','cd2','dfdcp','wilddf','ffiw']
    else:
        val_set_d = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/dfdc_test_lip.txt')
        val_loader_d = torch.utils.data.DataLoader(val_set_d, batch_size=batch_size, shuffle=True)
        val_set_cd1 = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/CD1_test.txt')
        val_loader_cd1 = torch.utils.data.DataLoader(val_set_cd1, batch_size=batch_size, shuffle=True)
        val_set_dfr = TestDataset(txt_path=r'F:/zhangjian/proj/PDI/data_2023/DFR_test.txt')
        val_loader_dfr = torch.utils.data.DataLoader(val_set_dfr, batch_size=batch_size, shuffle=True)
        val_loaders = [val_loader_ffpp,val_loader_cd1,val_loader_cd2,val_loader_d,val_loader_dp,val_loader_wdf,val_loader_dfr,val_loader_ffiw]
        val_names = ['ffpp','cd1','cd2','dfdc','dfdcp','wilddf','dfr','ffiw']

    net = MainModel(pretrained=True)
    device = torch.device("cuda:0")

    net = net.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    # optimizer
    optimizer = optim.AdamW(net.parameters(), lr=LR)
    #optimizer = SAM(net.parameters(),optim.SGD,lr=lr,momentum=0.9)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size= 100, gamma= 0.5)
    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, min_lr=1e-6,patience=2)
    with open(os.path.join(args.outf,"acc.txt"), "w") as f:
        with open(os.path.join(args.outf,"log.txt"), "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step(epoch)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                length = len(training_generator_df)
                for i, data in enumerate(zip(training_generator_df,training_generator_f2f,training_generator_fs,training_generator_nt,training_generator_or), 0):

                    data_df,data_f2f,data_fs,data_nt,data_or = data
                    input_df,target_df = data_df
                    input_f2f,target_f2f = data_f2f
                    input_fs,target_fs = data_fs
                    input_nt,target_nt = data_nt
                    input_or,target_or = data_or

                    input = torch.cat([input_df,input_f2f,input_fs,input_nt,input_or],dim=0)
                    target = torch.cat([target_df,target_f2f,target_fs,target_nt,target_or],dim=0)
                    input, target = input.to(device), target.to(device)
                    # ��
                    optimizer.zero_grad()
                    # forward + backward

                    # output = net(input)
                    net.model.reset_attn_drop_rate(0.1)
                    output,attn,feat0 = net(input,return_attn=True)
                    net.model.reset_attn_drop_rate(0.)
                    feat1,feat2 = net(input,aps=attn)

                    loss = criterion(output, target) + 0.1 * (mse_loss(feat1,feat0.detach())+mse_loss(feat2,feat0.detach()))
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()
                    print('[epoch:%d, iter:%d/%d] Lr: %.6f | Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), (epoch+1) * length, optimizer.param_groups[0]['lr'], sum_loss / (i + 1),
                             100. * float(correct) / float(total)))
                    f2.write('%03d  %05d/%d  Lr: %.6f |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), (epoch+1) * length, optimizer.param_groups[0]['lr'], sum_loss / (i + 1),
                                100. * float(correct) / float(total)))
                    f2.write('\n')
                    f2.flush()
                    # break
                scheduler.step(sum_loss/length)

                print("Waiting Test!")
                with torch.no_grad():
                    net.eval()
                    f.write("EPOCH=%03d:" % (epoch + 1))
                    for val_name,val_loader in zip(val_names,val_loaders):
                        correct = 0
                        total = 0
                        labels = []
                        pre = []
                        for data in tqdm(val_loader):
                            images,label = data
                            labels.extend(label)
                            images, label = images.to(device),  label.to(device)
                            outputs = net(images)

                            _, predicted = torch.max(outputs.data, 1)
                            total += label.size(0)
                            correct += (predicted == label).cpu().sum()
                            pre.extend(((F.softmax(outputs, dim=1)[:, 1]).cpu()).numpy())
                            # break
                        # print(pre)
                        acc = 100. * float(correct) / float(total)
                        auc = 100. * roc_auc_score(labels, pre)
                        print('%s|acc:%.3f%%, auc:%.3f%%' % (val_name,acc,auc))
                        f.write("%s|acc:%.3f%%, auc:%.3f%%; " % (val_name,acc,auc))

                    f.write('\n')
                    f.flush()
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

if __name__ == "__main__":
    seed = 111
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(only_cd=True)