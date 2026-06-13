import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def _gaussian_kernel_1d(sigma, device):
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum(), radius


def smooth_score_map(score_map, sigma):
    if sigma is None or sigma <= 0:
        return score_map
    kernel, radius = _gaussian_kernel_1d(sigma, score_map.device)
    m = score_map.unsqueeze(0).unsqueeze(0)
    m = F.conv2d(m, kernel.view(1, 1, 1, -1), padding=(0, radius))
    m = F.conv2d(m, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return m.squeeze(0).squeeze(0)


def image_anomaly_score(patch_scores, method='mean_top_k', top_k=10, smooth_sigma=0.0):
    if smooth_sigma and smooth_sigma > 0:
        side = int(round(patch_scores.numel() ** 0.5))
        if side * side == patch_scores.numel():
            smoothed = smooth_score_map(patch_scores.view(side, side), smooth_sigma)
            patch_scores = smoothed.reshape(-1)
    if method == 'max':
        return patch_scores.max()
    k = min(top_k, patch_scores.numel())
    return patch_scores.topk(k).values.mean()


def compute_threshold(calib_scores, method='quantile', param=0.99):
    """Pick a decision threshold from held-out GOOD scores only (leakage-free).

    method='sigma'    : mean + param*std            (param = k, e.g. 3.0)
    method='quantile' : the param-quantile of good   (param = q in [0,1];
                        q=0.99 -> tolerate ~1% false positives on good)
    method='max'      : max good score               (most conservative)

    The threshold is NEVER chosen using test labels — that would be data
    leakage. We only set an operating point on the good distribution.
    """
    s = np.asarray(calib_scores, dtype=np.float64)
    if method == 'sigma':
        return float(s.mean() + param * s.std())
    if method == 'quantile':
        return float(np.quantile(s, param))
    if method == 'max':
        return float(s.max())
    raise ValueError(f"Unknown threshold method: {method}")


class Config:
    def __init__(self):
        self.base_path = Path('mvtec_anomaly_detection')
        self.dataset_name = 'toothbrush'
        self.output_root = Path('outputs')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4
        self.num_workers = 2
        self.image_size = 224
        self.feature_dim = 1536
        self.coreset_ratio = 0.1
        self.score_method = 'max'
        self.score_top_k = 10
        self.smooth_sigma = 1.0
        self.threshold_method = 'sigma'      # 'sigma' | 'quantile' | 'max'
        self.threshold_param = 2.0           # sigma: threshold = mean + k*std on held-out good
                                             #   k=2.0 -> ~2.3% expected FPR (Gaussian); robust on
                                             #   small calib sets. (quantile: param = q in [0,1].)
        
    def get_paths(self):
        return {
            'train_good': self.base_path / self.dataset_name / 'train' / 'good',
            'test_good': self.base_path / self.dataset_name / 'test' / 'good',
            'test_defective': self.base_path / self.dataset_name / 'test' / 'defective',
        }


class ImageTransform:
    def __init__(self, image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std))
        ])

    def __call__(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super(ResNetFeatureExtractor, self).__init__()
        self.device = device
        
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device)
        
        self.features = []
        
        def hook(module, input, output):
            self.features.append(output)
        
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
    
    def forward(self, x):
        self.features = []
        x = x.to(self.device)
        
        with torch.no_grad():
            _ = self.model(x)
        
        avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        fmap_size = self.features[0].shape[-2]
        adaptive_pool = nn.AdaptiveAvgPool2d(fmap_size)
        
        resized_maps = [adaptive_pool(avg_pool(fmap)) for fmap in self.features]
        
        patch = torch.cat(resized_maps, dim=1)
        patch = patch.reshape(patch.shape[1], -1).T
        
        return patch


class ViTFeatureExtractor(nn.Module):
    def __init__(self, device='cuda', model_name='vit_small_patch8_224.dino', img_size=None):
        super(ViTFeatureExtractor, self).__init__()
        self.device = device

        import timm
        from timm.data import resolve_model_data_config

        kwargs = dict(pretrained=True)
        if img_size is not None:
            kwargs['img_size'] = img_size
            kwargs['dynamic_img_size'] = True

        self.model = timm.create_model(model_name, **kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device)

        cfg = resolve_model_data_config(self.model)
        size = img_size or cfg['input_size'][-1]
        self.transform = ImageTransform(image_size=size, mean=cfg['mean'], std=cfg['std'])
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 1)

    def forward(self, x):
        x = x.to(self.device)

        with torch.no_grad():
            tokens = self.model.forward_features(x)

        patch = tokens[:, self.num_prefix_tokens:, :].squeeze(0)

        return patch


def get_feature_extractor(name, device='cuda'):
    if name == 'resnet50':
        extractor = ResNetFeatureExtractor(device=device)
        extractor.transform = ImageTransform(image_size=224)
        return extractor
    if name == 'dino_vit':
        return ViTFeatureExtractor(device=device, model_name='vit_small_patch8_224.dino')
    if name == 'dinov2':
        return ViTFeatureExtractor(device=device, model_name='vit_small_patch14_dinov2.lvd142m')
    raise ValueError(f"Unknown backbone: {name}")


class MemoryBank:
    def __init__(self, device='cuda'):
        self.device = device
        self.bank = None
        self.size = 0
    
    def build(self, features_list):
        bank = torch.cat(features_list, dim=0)
        bank = bank / (torch.norm(bank, dim=1, keepdim=True) + 1e-8)
        self.bank = bank.to(self.device)
        self.size = self.bank.shape[0]
    
    def k_center_greedy_select(self, ratio=0.1, max_pool=50000):
        if self.size > max_pool:
            idx = torch.linspace(0, self.size - 1, max_pool).long().to(self.device)
            self.bank = self.bank[idx]
            self.size = self.bank.shape[0]

        k = max(1, int(self.size * ratio))

        if k >= self.size:
            return self.bank

        selected_idx = [0]
        min_distances = torch.cdist(self.bank, self.bank[0:1]).squeeze(1)
        min_distances[0] = -1

        for _ in range(1, k):
            farthest_idx = torch.argmax(min_distances).item()
            selected_idx.append(farthest_idx)

            new_dists = torch.cdist(self.bank, self.bank[farthest_idx:farthest_idx+1]).squeeze(1)
            min_distances = torch.minimum(min_distances, new_dists)
            min_distances[farthest_idx] = -1

        selected_bank = self.bank[selected_idx]
        self.bank = selected_bank
        self.size = selected_bank.shape[0]

        return self.bank
    
    def get(self):
        return self.bank


class NormalizedFeatureExtractor:
    def __init__(self, backbone, device='cuda'):
        self.backbone = backbone
        self.device = device
    
    def extract_and_normalize(self, image_tensor):
        features = self.backbone(image_tensor)
        features_normalized = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
        return features_normalized


class AnomalyDetector:
    def __init__(self, backbone, memory_bank, device='cuda'):
        self.backbone = backbone
        self.memory_bank = memory_bank
        self.device = device
        self.threshold = None
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def compute_anomaly_score(self, features):
        distances = torch.cdist(features, self.memory_bank.get(), p=2.0)
        anomaly_scores = torch.min(distances, dim=1)[0]
        return anomaly_scores
    
    def predict(self, image_tensor):
        with torch.no_grad():
            features = self.backbone(image_tensor)
            features_normalized = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
            anomaly_scores = self.compute_anomaly_score(features_normalized)
        
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        predictions = (anomaly_scores > self.threshold).long()
        return {
            'anomaly_scores': anomaly_scores.cpu().numpy(),
            'predictions': predictions.cpu().numpy(),
            'features': features.cpu().numpy()
        }
