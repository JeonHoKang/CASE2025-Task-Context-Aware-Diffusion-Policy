from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import torch
import torch.nn as nn
import torchvision


class FiLM(nn.Module):
    def __init__(self, clip_dim, feature_dim):
        """
        FiLM layer for feature-wise modulation using CLIP embeddings.
        clip_dim: Dimension of CLIP text embeddings (e.g., 512 or 768)
        feature_dim: Number of feature channels in ResNet to modulate
        """
        super(FiLM, self).__init__()

        # Project CLIP embedding to match ResNet feature dimensions
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(clip_dim, feature_dim * 2),  # Predicts γ and β
            nn.Unflatten(-1, (2, feature_dim, 1, 1))  # Reshapes for broadcasting
        )

    def forward(self, x, clip_emb):
        """
        x: Feature maps from ResNet layers -> Shape: (batch_size, channels, H, W)
        clip_emb: CLIP text embedding -> Shape: (batch_size, clip_dim)
        """
        batch_size, channels, height, width = x.shape 
        film_params = self.cond_encoder(clip_emb)  # Get γ and β from CLIP embedding
        film_params = film_params.view(batch_size, 2, channels, 1, 1)  # Expand for broadcasting

        gamma = film_params[:, 0, ...]  # First half is scale
        beta = film_params[:, 1, ...]   # Second half is shift
         
        return (gamma * x) + beta  # Apply FiLM modulation
    
class FiLMResidualBlock(nn.Module):
    def __init__(self, original_block, language_dim):
        """
        Wraps a ResNet residual block and applies FiLM before every BatchNorm layer.
        """
        super(FiLMResidualBlock, self).__init__()
        self.original_block = original_block
        feature_dim = original_block.conv1.out_channels  # Get feature dimension
        self.film1 = FiLM(language_dim, feature_dim)
        self.film2 = FiLM(language_dim, feature_dim)

    def forward(self, x, lang_emb):
        identity = x  # <-- Skip connection (original input)

        # First conv layer with FiLM
        out = self.original_block.conv1(x)
        out = self.original_block.bn1(out)
        out = self.film1(out, lang_emb)  # Apply FiLM
        out = self.original_block.relu(out)

        # Second conv layer with FiLM
        out = self.original_block.conv2(out)
        out = self.original_block.bn2(out)
        out = self.film2(out, lang_emb)  # Apply FiLM

        # Apply skip connection (residual sum)
        if self.original_block.downsample is not None:
            identity = self.original_block.downsample(x)  # Adjust dimensions if needed

        out += identity  # <-- This is the skip connection!
        out = self.original_block.relu(out)
        return out
    
class ResNetWithFiLM(nn.Module):
    def __init__(self, resnet, language_dim):
        super(ResNetWithFiLM, self).__init__()
        
        self.language_dim = language_dim
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Wrap all ResNet layers with FiLM
        self.layer1 = self.wrap_with_film(resnet.layer1, language_dim)
        self.layer2 = self.wrap_with_film(resnet.layer2, language_dim)
        self.layer3 = self.wrap_with_film(resnet.layer3, language_dim)
        self.layer4 = self.wrap_with_film(resnet.layer4, language_dim)
        self.feature_dim = language_dim  # ✅ Store the correct feature dimension

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(self.feature_dim, 512)  # ✅ Use stored feature_dim

    def wrap_with_film(self, layer, language_dim):
        return nn.ModuleList([FiLMResidualBlock(block, language_dim) for block in layer])

    def forward(self, x, lang_emb):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Apply FiLM-conditioned residual blocks
        for block in self.layer1:
            x = block(x, lang_emb)
        for block in self.layer2:
            x = block(x, lang_emb)
        for block in self.layer3:
            x = block(x, lang_emb)
        for block in self.layer4:
            x = block(x, lang_emb)

        x = self.avgpool(x).flatten(1)
        return self.fc(x)
    
class train_utils:
    def __init__(self):
        pass

    #@markdown ### **Vision Encoder**
    #@markdown
    #@markdown Defines helper functions:
    #@markdown - `get_resnet` to initialize standard ResNet vision encoder
    #@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

    def get_resnet_FILM(self, name, weights="ResNet34_Weights.IMAGENET1K_V1", fine_tune = True, **kwargs):
        """
        name: resnet18, resnet34, resnet50
        weights: "IMAGENET1K_V1", "r3m"
        """
        # load r3m weights
        if (weights == "r3m") or (weights == "R3M"):
            return self.get_r3m(name=name, **kwargs)

        func = getattr(torchvision.models, name)
        
        resnet = func(weights=weights, **kwargs)
        if fine_tune:
            resnet.requires_grad = True
        feature_dim = resnet.fc.in_features  # Store this before modifying fc
        resnet.fc = torch.nn.Identity()  # Remove classification head
        return ResNetWithFiLM(resnet, feature_dim)
    
    def get_resnet(self, name, weights="ResNet34_Weights.IMAGENET1K_V1", fine_tune = True, **kwargs):
        """
        name: resnet18, resnet34, resnet50
        weights: "IMAGENET1K_V1", "r3m"
        """
        # load r3m weights
        if (weights == "r3m") or (weights == "R3M"):
            return self.get_r3m(name=name, **kwargs)

        func = getattr(torchvision.models, name)
        
        resnet = func(weights=weights, **kwargs)
        if fine_tune:
            resnet.requires_grad = True
        resnet.fc = torch.nn.Identity()  # Remove classification head
        return resnet

    def get_r3m(self, name, **kwargs):
        """
        name: resnet18, resnet34, resnet50
        """
        import r3m
        r3m.device = 'cpu'
        model = r3m.load_r3m(name)
        r3m_model = model.module
        resnet_model = r3m_model.convnet
        resnet_model = resnet_model.to('cpu')
        return resnet_model

    def replace_submodules(self,
            root_module: nn.Module,
            predicate: Callable[[nn.Module], bool],
            func: Callable[[nn.Module], nn.Module]) -> nn.Module:
        """
        Replace all submodules selected by the predicate with
        the output of func.

        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
        """
        if predicate(root_module):
            return func(root_module)

        bn_list = [k.split('.') for k, m
            in root_module.named_modules(remove_duplicate=True)
            if predicate(m)]
        for *parent, k in bn_list:
            parent_module = root_module
            if len(parent) > 0:
                parent_module = root_module.get_submodule('.'.join(parent))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all modules are replaced
        bn_list = [k.split('.') for k, m
            in root_module.named_modules(remove_duplicate=True)
            if predicate(m)]
        assert len(bn_list) == 0
        return root_module

    def replace_bn_with_gn(self,
        root_module: nn.Module,
        features_per_group: int=16) -> nn.Module:
        """
        Relace all BatchNorm layers with GroupNorm.
        """
        self.replace_submodules(
            root_module=root_module,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//features_per_group,
                num_channels=x.num_features)
        )
        return root_module
    
class SimpleViTEncoder(nn.Module):
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True, frozen: bool = False):
        super().__init__()
        # Load the ViT model
        self.vision_encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # Remove classifier

        # Optionally freeze the model if required
        if frozen:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Pass the input through the ViT model
        return self.vision_encoder(x)
    
    
import timm
import numpy as np
import copy

class TransformerObsEncoder(nn.Module):
    def __init__(self,
                 shape_meta: dict,
                 model_name: str = 'vit_base_patch16_clip_224.openai',
                 global_pool: str = '',
                 transforms: list = None,
                 n_emb: int = 768,
                 pretrained: bool = True,
                 frozen: bool = False,
                 use_group_norm: bool = True,
                 share_rgb_model: bool = False,
                 feature_aggregation: str = None,
                 downsample_ratio: int = 32):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool,  # '' means no pooling
            num_classes=0  # remove classification layer
        )
        self.model_name = model_name

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        feature_dim = None
        if model_name.startswith('resnet'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = self.replace_batch_norm_with_group_norm(model)

        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            if self.feature_aggregation is None:
                pass
            elif self.feature_aggregation != 'cls':
                print(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'

        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model

                with torch.no_grad():
                    example_img = torch.zeros((1,) + tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]

                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                dim = np.prod(shape)
                proj = nn.Identity()
                if dim != n_emb:
                    proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj

                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]
            assert self.feature_aggregation is None
            return feature

        assert len(feature.shape) == 4
        feature = torch.flatten(feature, start_dim=-2)  # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2)  # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature

    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B, -1, self.n_emb)
            embeddings.append(emb)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B, T, -1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)

        result = torch.cat(embeddings, dim=1)
        return result

    def replace_batch_norm_with_group_norm(self, model):
        def replace_submodules(root_module, predicate, func):
            for name, module in root_module.named_children():
                if predicate(module):
                    root_module.add_module(name, func(module))
                else:
                    replace_submodules(module, predicate, func)
            return root_module

        return replace_submodules(
            root_module=model,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                num_channels=x.num_features
            )
        )

def test():
    shape_meta = {
        'obs': {
            'rgb': {'shape': (3, 224, 224), 'type': 'rgb'},
            'low_dim': {'shape': (10,), 'type': 'low_dim'}
        }
    }
    
    encoder = TransformerObsEncoder(shape_meta=shape_meta)
    obs_dict = {
        'rgb': torch.rand(2, 5, 3, 224, 224),
        'low_dim': torch.rand(2, 5, 10)
    }
    
    result = encoder(obs_dict)
    print(result.shape)
    
# test()