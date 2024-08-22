import torch
import torch.nn as nn
import math
# import ListConfig
import torch.nn.functional as F
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, l1_loss, log_softmax
import pytorch_lightning as pl
from omegaconf import OmegaConf
import sys
import pickle

def positional_encoding_from_cfg(cfg, feature_dim): 
    # if cfg.positional_encoding.type == 'PeriodicPositionalEncoding': 
    #     return PeriodicPositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
    # el
    if cfg.positional_encoding.type == 'PositionalEncoding':
        return PositionalEncoding(feature_dim, **cfg.positional_encoding)
    elif cfg.positional_encoding.type == 'LearnedPositionEmbedding':
        return LearnedPositionEmbedding(dim=feature_dim, **cfg.positional_encoding)
    elif not cfg.positional_encoding.type or str(cfg.positional_encoding.type).lower() == 'none':
        return None
    raise ValueError("Unsupported positional encoding")

def init_alibi_biased_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper but 
    with not with the future masked out.
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n)) 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

class TransformerEncoder(torch.nn.Module):

    def __init__(self, cfg, input_dim) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = input_dim

        self._init_transformer()
        # self._init_biased_mask()
        self.biased_mask = init_alibi_biased_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        


    def _init_transformer(self):
        self.bottleneck = nn.Linear(self.input_dim, self.cfg.feature_dim)
        self.PE = positional_encoding_from_cfg(self.cfg, self.cfg.feature_dim) #None
        dim_factor = self._total_dim_factor()
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=self.cfg.feature_dim * dim_factor, 
                    nhead=self.cfg.nhead, 
                    dim_feedforward=dim_factor*self.cfg.feature_dim, 
                    activation=self.cfg.activation,
                    dropout=self.cfg.dropout, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.num_layers)
        # self.decoder = nn.Linear(dim_factor*self.input_dim, self.decoder_output_dim())

    # def _init_biased_mask(self):
    #     self.temporal_bias_type = self.cfg.get('temporal_bias_type', 'none')
    #     if self.temporal_bias_type == 'alibi':
    #         self.biased_mask = init_alibi_biased_mask(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
    #     elif self.temporal_bias_type == 'alibi_future':
    #         self.biased_mask = init_alibi_biased_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        # elif self.temporal_bias_type == 'faceformer':
        #     self.biased_mask = init_faceformer_biased_mask(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len, period=self.cfg.period)
        # elif self.temporal_bias_type == 'faceformer_future':
        #     self.biased_mask = init_faceformer_biased_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len, period=self.cfg.period)
        # elif self.temporal_bias_type == 'classic':
        #     self.biased_mask = init_mask(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        # elif self.temporal_bias_type == 'classic_future':
        #     self.biased_mask = init_mask_future(num_heads = self.cfg.nhead, max_seq_len = self.cfg.max_len)
        # elif self.temporal_bias_type == 'none':
        #     self.biased_mask = None
        # else:
        #     raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

    def encoder_input_dim(self):
        return self.input_dim

    def encoder_output_dim(self):
        return self.cfg.feature_dim

    def forward(self, sample, train=False, teacher_forcing=True): 
        if self.bottleneck is not None:
            sample["hidden_feature"] = self.bottleneck(sample["hidden_feature"])
        hidden_states = self._positional_enc(sample)
        encoded_feature = self._encode(sample, hidden_states)
        sample["encoded_sequence_feature"] = encoded_feature
        return sample
       
    def _pe_dim_factor(self):
        dim_factor = 1
        if self.PE is not None: 
            dim_factor = self.PE.output_size_factor()
        return dim_factor

    def _total_dim_factor(self): 
        return  self._pe_dim_factor()

    def _positional_enc(self, sample): 
        hidden_states = sample["hidden_feature"] 
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        return hidden_states

    def _encode(self, sample, hidden_states):
        if self.biased_mask is not None: 
            mask = self.biased_mask[:, :hidden_states.shape[1], :hidden_states.shape[1]].clone() \
                .detach().to(device=hidden_states.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(hidden_states.shape[0], 1, 1)
        else: 
            mask = None
        encoded_feature = self.encoder(hidden_states, mask=mask)
        B, T = encoded_feature.shape[:2]
        encoded_feature = encoded_feature.view(B*T, -1)
        encoded_feature = self.encoder(hidden_states)
        encoded_feature = encoded_feature.view(B, T, -1)
        return encoded_feature

class TransformerPooler(nn.Module):
    """
    inspired by: 
    https://huggingface.co/transformers/v3.3.1/_modules/transformers/modeling_bert.html#BertPooler 
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, sample, input_key = "encoded_sequence_feature"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = sample[input_key]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        sample["pooled_sequence_feature"] = pooled_output
        return sample

class SequenceClassificationEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.parameters())
        return []

    def output_feature_dim(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

class TransformerSequenceClassifier(SequenceClassificationEncoder):

    def __init__(self, cfg, input_dim, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        # self.num_classes = num_classes

        # self.transformer_encoder = transformer_encoder_from_cfg(cfg.encoder, input_dim)
        self.transformer_encoder = TransformerEncoder(cfg.encoder, input_dim)
        self.pooler = TransformerPooler(self.transformer_encoder.encoder_output_dim(), self.transformer_encoder.encoder_output_dim())
        # self.classifier = nn.Linear(self.transformer_encoder.encoder_output_dim(), self.num_classes)

    def encoder_output_dim(self):
        return self.transformer_encoder.encoder_output_dim()

    def forward(self, sample):
        sample = self.transformer_encoder(sample)
        sample = self.pooler(sample)
        # sample = self.classifier(sample)
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())

class ClassificationHead(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.parameters())
        return []

    def num_classes(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

class MultiheadLinearClassificationHead(ClassificationHead):
    
    def __init__(self, cfg, input_dim, num_classes):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        # assert isinstance(num_classes, (list, ListConfig))
        classification_heads = [LinearClassificationHead(cfg, input_dim, classes) for classes in num_classes]
        category_names=cfg.get('category_names', None)
        if category_names is None:
            head_names = [f"category_{i}" for i in range(len(num_classes))]
        else:
            head_names = category_names
        self.classification_heads = nn.ModuleDict(dict(zip(head_names, classification_heads)))

    def forward(self, sample, input_key="pooled_sequence_feature", output_key="predicted_logits"):
        for key, head in self.classification_heads.items():
            sample = head(sample, input_key, output_key + f"_{key}")
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())

class LinearClassificationHead(ClassificationHead): 
    def __init__(self, cfg, input_dim,  num_classes):
        super().__init__()
        self.cfg = cfg
        # self.input_dim = cfg.input_dim
        self.input_dim = input_dim
        # self.num_classes = cfg.num_classes
        self.num_classes = num_classes

        self.dropout = nn.Dropout(cfg.dropout_prob)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, sample, input_key="pooled_sequence_feature", output_key="predicted_logits"):
        sample[output_key] = self.classifier(sample[input_key])
        return sample

    def get_trainable_parameters(self):
        return list(self.parameters())

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if not self.include_top:
            return x

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")

def loss_from_cfg(config, loss_name):
    if loss_name in config.keys():
        if isinstance(config[loss_name], str):
            loss = class_from_str(config[loss_name], sys.modules[__name__])
        else:
            cont = OmegaConf.to_container(config[loss_name])
            if isinstance(cont, list):
                loss = {name: 1. for name in cont}
            elif isinstance(cont, dict):
                loss = cont
            else:
                raise ValueError(f"Unkown type of loss '{type(cont)}' for loss '{loss_name}'")
    else:
        loss = None
    return loss

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

class EmotionRecognitionBaseModule(pl.LightningModule):
    """
    EmotionRecognitionBaseModule is a base class for emotion prediction (valence and arousal, expression classification and/or action units)
    """

    def __init__(self, config):
        """

        """
        super().__init__()
        self.config = config
        # properties
        self.predicts_expression = self.config.model.predict_expression
        self.predicts_valence = self.config.model.predict_valence
        self.predicts_arousal = self.config.model.predict_arousal
        if 'predict_AUs' in self.config.model.keys() and self.config.model.predict_AUs:
            self.predicts_AUs = self.config.model.predict_AUs
        else :
            self.predicts_AUs = 0

        # if 'v_activation' in config.model.keys():
        #     self.v_activation = class_from_str(self.config.model.v_activation, sys.modules[__name__])
        # else:
        self.v_activation = None

        # if 'a_activation' in config.model.keys():
        #     self.a_activation = class_from_str(self.config.model.a_activation, sys.modules[__name__])
        # else:
        self.a_activation = None

        # if 'exp_activation' in config.model.keys():
        #     self.exp_activation = class_from_str(self.config.model.exp_activation, sys.modules[__name__])
        # else:
        self.exp_activation = F.log_softmax

        # if 'AU_activation' in config.model.keys():
        #     self.AU_activation = class_from_str(self.config.model.AU_activation, sys.modules[__name__])
        # else:
        self.AU_activation = None

        self.va_loss = loss_from_cfg(config.model, 'va_loss')
        self.v_loss = loss_from_cfg(config.model, 'v_loss')
        self.a_loss = loss_from_cfg(config.model, 'a_loss')
        self.exp_loss = loss_from_cfg(config.model, 'exp_loss')
        self.AU_loss = loss_from_cfg(config.model, 'AU_loss') # None

    def forward(self, image):
        raise NotImplementedError()

class EmoCnnModule(EmotionRecognitionBaseModule):
    """
    Emotion Recognitition module which uses a conv net as its backbone. Currently Resnet-50 and VGG are supported. 
    ResNet-50 based emotion recognition trained on AffectNet is the network used for self-supervising emotion in EMOCA.
    """
    def __init__(self, config):
        super().__init__(config)
        # self.n_expression = 9  # we use all affectnet classes (included none) for now
        self.n_expression = config.data.n_expression if 'n_expression' in config.data.keys() else 9

        self.num_outputs = 0
        if self.predicts_expression:
            self.num_outputs += self.n_expression
            self.num_classes = self.n_expression

        if self.predicts_valence:
            self.num_outputs += 1

        if self.predicts_arousal:
            self.num_outputs += 1

        if self.predicts_AUs:
            self.num_outputs += self.config.model.predict_AUs

        if config.model.backbone == "resnet50":
            # self.backbone = resnet50(num_classes=8631, include_top=False)
            self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=8631, include_top=False)

            self.last_feature_size = 2048
            self.linear = nn.Linear(self.last_feature_size, self.num_outputs) # 2048 is the output of  the resnet50 backbone without the MLP "top"
        # elif config.model.backbone[:3] == "vgg":
        #     vgg_constructor = getattr(vgg, config.model.backbone)
        #     self.backbone = vgg_constructor(pretrained=bool(config.model.load_pretrained), progress=True)
        #     self.last_feature_size = 1000
        #     self.linear = Linear(self.last_feature_size, self.num_outputs) #1000 is the number of imagenet classes so the dim of the output of the vgg backbone
        # else:
        #     raise ValueError(f"Invalid backbone: '{self.config.model.backbone}'")

    def get_last_feature_size(self):
        return self.last_feature_size

    def _forward(self, images):
        output = self.backbone(images) #(BS*T,2048,1,1)
        emo_feat_2 = output
        output = self.linear(output.view(output.shape[0], -1)) #(BS*T,)

        out_idx = 0
        if self.predicts_expression:
            expr_classification = output[:, out_idx:(out_idx + self.n_expression)]
            if self.exp_activation is not None:
                expr_classification = self.exp_activation(expr_classification, dim=1)
            out_idx += self.n_expression
        else:
            expr_classification = None

        if self.predicts_valence:
            valence = output[:, out_idx:(out_idx + 1)]
            if self.v_activation is not None:
                valence = self.v_activation(valence)
            out_idx += 1
        else:
            valence = None

        if self.predicts_arousal:
            arousal = output[:, out_idx:(out_idx + 1)]
            if self.a_activation is not None:
                arousal = self.a_activation(arousal)
            out_idx += 1
        else:
            arousal = None


        if self.predicts_AUs:
            num_AUs = self.config.model.predict_AUs
            AUs = output[:, out_idx:(out_idx + num_AUs)]
            if self.AU_activation is not None:
                AUs = self.AU_activation(AUs)
            out_idx += num_AUs
        else:
            AUs = None

        assert out_idx == output.shape[1]

        values = {}
        values["emo_feat_2"] = emo_feat_2
        values["valence"] = valence
        values["arousal"] = arousal
        values["expr_classification"] = expr_classification
        values["AUs"] = AUs
        return values


    def forward(self, batch):
        images = batch['image']

        if len(images.shape) == 5:
            K = images.shape[1]
        elif len(images.shape) == 4:
            K = 1
        else:
            raise RuntimeError("Invalid image batch dimensions.")

        # print("Batch size!")
        # print(images.shape)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        emotion = self._forward(images)

        valence = emotion['valence']
        arousal = emotion['arousal']

        # emotion['expression'] = emotion['expression']

        # classes_probs = F.softmax(emotion['expression'])
        # expression = self.exp_activation(emotion['expr_classification'], dim=1)

        values = {}
        if self.predicts_valence:
            values['valence'] = valence.view(-1,1)
        if self.predicts_arousal:
            values['arousal'] = arousal.view(-1,1)
        # values['expr_classification'] = expression
        values['expr_classification'] = emotion['expr_classification']
        if self.predicts_AUs:
            values['AUs'] = emotion['AUs']

        values["emo_feat_2"] = emotion["emo_feat_2"]

        # TODO: WARNING: HACK
        if 'n_expression' not in self.config.data:
            if self.n_expression == 8:
                raise NotImplementedError("This here should not be called")
                values['expr_classification'] = torch.cat([
                    values['expr_classification'], torch.zeros_like(values['expr_classification'][:, 0:1])
                                                   + 2*values['expr_classification'].min()],
                    dim=1)

        return values

    def _get_trainable_parameters(self):
        return list(self.backbone.parameters())
