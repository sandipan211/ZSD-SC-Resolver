
from cls_models import ClsUnseenTrain,Regressor
from generate import load_seen_att
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from mmdetection.splits import get_seen_class_ids
import sys,os




# %psource ClsUnseenTrain.forward

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

opt = dotdict({
    'dataset':'voc',
    'classes_split': '16_4',
    'class_embedding': 'VOC/fasttext_synonym.npy',
    'dataroot':'/workspace/arijit_ug/sushil/zsd/data/pascal/feat',
    'trainsplit': 'train_0.6_0.3',
    
})
# path to save the trained classifier best checkpoint
path = 'VOC/voc_regressor_s100.pth'
#loading seen attributes and labels from the semantic reprsentation
seen_att, att_labels = load_seen_att(opt)
classid_tolabels = {l:i for i, l in enumerate(att_labels.data.numpy())}

#print(classid_tolabels)
# print("seen attr ")
# print(seen_att)
# print(seen_att.shape)
# print(att_labels)
# print(att_labels.shape,att_labels.dtype)

#all_sem_att=np.load(opt.class_embedding) #loading all the embedding




print("training regressor in process......")
seen_features = np.load(f"{opt.dataroot}/{opt.trainsplit}_feats.npy")
seen_labels = np.load(f"{opt.dataroot}/{opt.trainsplit}_labels.npy")
print(seen_features.shape)
#print(seen_features)

###for test_index
# seen_features_test = np.load(f"{opt.dataroot}/{opt.trainsplit}_feats_5.npy")
# seen_labels_test = np.load(f"{opt.dataroot}/{opt.trainsplit}_labels_5.npy")
# inds_test = np.random.permutation(np.arange(len(seen_labels_test)))
# total_test_examples=int(1*len(seen_labels_test))
# test_inds = inds_test[:total_test_examples]
# ####
inds = np.random.permutation(np.arange(len(seen_labels)))
total_train_examples = int (0.7 * len(seen_labels))
train_inds = inds[:total_train_examples]
test_inds = inds[total_train_examples:]

#len(test_inds)+len(train_inds), len(seen_labels)
#print(seen_labels[1:1000])


train_feats = seen_features[train_inds]
train_labels = seen_labels[train_inds]
test_feats = seen_features[test_inds]
test_labels = seen_labels[test_inds]
#print(test_labels)

# bg_inds = np.where(seen_labels==0)
# fg_inds = np.where(seen_labels>0)

# a=seen_features.shape[1]
# b=seen_att.shape[1]
# print(a)
# print(b)



regressor_seen = Regressor().cuda()
print('network structure :\n',regressor_seen)



class Featuresdataset(Dataset):
     
    def __init__(self, features, labels, classid_tolabels):
        self.classid_tolabels = classid_tolabels
        self.features = features
        self.labels = labels
        

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
#         import pdb; pdb.set_trace()
        
        if self.classid_tolabels is not None:
            batch_label = self.classid_tolabels[batch_label]
        return batch_feature, batch_label

    def __len__(self):
        return len(self.labels)

#seen_labels.shape

dataset_train = Featuresdataset(train_feats, train_labels, classid_tolabels)
dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True) 
dataset_test = Featuresdataset(test_feats, test_labels, classid_tolabels)
dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=True)
###############################################

#setting optimizer and criterion
from torch.optim.lr_scheduler import StepLR
##awa
# lr=9.9945e-5
# decay = 2.645e-5
# weight_decay =  0
##cub
# lr=9.9978e-5
# decay = 8.75e-5
# weight_decay =  0
#sun
lr=9.6469e-5
decay = 1.38e-2
weight_decay =  1e-3
# ##flo
# lr=9.99e-5
# decay = 2.28e-5
# weight_decay =  0

beta1 =  0.9
beta2 = 0.999
optimizer=optim.Adam(regressor_seen.parameters(), lr = lr, weight_decay = weight_decay, betas = (beta1, beta2))

lr_lambda = lambda global_step: 1/(1 + global_step*decay)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.MSELoss(reduction='sum') #using (squared L2 norm)
#criterion = nn.MSELoss(reduction='mean')
######################################################
min_val_loss = float("inf")  #willl used in saving appropriate model


#validation function
def val():
    running_loss = 0.0
    global min_val_loss
    regressor_seen.eval()
    with torch.no_grad():
        
        for i, (inputs, labels) in enumerate(dataloader_test, 0):
            #semantic_true=form_semantic_batch(labels,1024)
            semantic_true=seen_att[labels]
            #print(semantic_true.dtype)
            inputs = inputs.cuda()
            #labels = labels.cuda()
            semantic_true=semantic_true.cuda()

            semantic_pred = regressor_seen(inputs)
            loss = criterion(semantic_pred.float(), semantic_true.float())

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Validation Loss {epoch + 1},[{i + 1} / {len(dataloader_test)}], total_loss:{(running_loss / i) :0.4f}')
                #f.write(f'Validation Loss {epoch + 1}, [{i + 1} / {len(dataloader_test)}], total_loss:{(running_loss / i) :0.5f}\n')
        if (running_loss / i) < min_val_loss:
            min_val_loss = running_loss / i
            state_dict = regressor_seen.state_dict()   
            torch.save(state_dict, path)
            print(f'saved {min_val_loss :0.4f}')
            #f.write(f'saved {min_val_loss :0.4f}')

#training code 

for epoch in range(100):
    regressor_seen.train() #training mode
    running_loss = 0.0 #for calculting overall loss
    
   
    for i, (inputs, labels) in enumerate(dataloader_train, 0):
        
        semantic_true=seen_att[labels].cuda()
        inputs = inputs.cuda()
        #labels = labels.cuda()
        #semantic_true=semantic_true.cuda()
        
        optimizer.zero_grad()

        semantic_pred = regressor_seen(inputs)
        #loss = criterion(outputs, labels)
        
        
        loss=criterion(semantic_pred.float(),semantic_true.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999: 
            print(f'Train Loss {epoch + 1}, [{i + 1} / {len(dataloader_train)}], total_loss:{(running_loss / i) :0.5f}')
            #f.write(f'Train Loss {epoch + 1}, [{i + 1} / {len(dataloader_train)}], total_loss:{(running_loss / i) :0.5f}\n')
    val()
    lr_scheduler.step()
    
print('Finished Training')

#if __name__ == '__main__':
#    main()