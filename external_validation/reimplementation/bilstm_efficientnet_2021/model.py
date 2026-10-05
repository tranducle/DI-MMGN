import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SpatialDropout1D(nn.Module):
    def __init__(self,p=0.2):
        super().__init__(); self.p=float(p)
    def forward(self,x):
        if not self.training or self.p<=0: return x
        keep=1.0-self.p
        mask=torch.empty((x.size(0),1,x.size(2)),device=x.device,dtype=x.dtype).bernoulli_(keep)/keep
        return x*mask

class TextBiLSTM2021(nn.Module):
    def __init__(self,vocab_size,embed_dim=64,hidden_per_direction=64,spatial_dropout=0.2):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.sdrop=SpatialDropout1D(spatial_dropout)
        self.lstm=nn.LSTM(embed_dim,hidden_per_direction,batch_first=True,bidirectional=True)
        self.classifier=nn.Linear(hidden_per_direction*2,2)
    def features(self,ids):
        x=self.sdrop(self.embedding(ids))
        out,(h,c)=self.lstm(x)
        feat=torch.cat([h[-2],h[-1]],dim=-1)
        return feat
    def forward(self,ids):
        return self.classifier(self.features(ids))

class EfficientNetB02021(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net=efficientnet_b0(weights=weights)
        self.features_net=net.features
        self.avgpool=net.avgpool
        self.bn0=nn.BatchNorm1d(1280)
        self.fc=nn.Linear(1280,128)
        self.bn1=nn.BatchNorm1d(128)
        self.out=nn.Linear(128,2)
    def features(self,x):
        x=self.features_net(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.bn0(x)
        x=self.fc(x)
        x=self.bn1(x)
        return x
    def forward(self,x):
        return self.out(self.features(x))

def soft_vote_probs(text_logits,image_logits):
    return (torch.softmax(text_logits,dim=-1)+torch.softmax(image_logits,dim=-1))/2.0
