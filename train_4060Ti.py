import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from einops import rearrange, repeat
import platform

import torch.utils.data as data
from PIL import Image
import numpy as np

# 开启 TF32 (Ampere/Ada 架构核心加速)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# # 导入新版 autocast
# try:
#     from torch.amp import autocast
# except ImportError:
#     from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.npy', '.NPY', '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def array_loader(path):
    # Load native arrays or TIFFs
    if path.endswith('.npy') or path.endswith('.NPY'):
        return np.load(path)
    else:
        # Fallback for TIFFs
        from PIL import Image
        return np.array(Image.open(path))

class DenoisingDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[256, 256]):
        self.data_root = data_root
        self.image_size = image_size
        self.max_log_val = 7.0
        
        # Point to the ground-truth folder
        gt_dir = os.path.join(data_root, 'gt')
        flist = make_dataset(gt_dir)
        
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist

    def __getitem__(self, index):
        ret = {}
        gt_path = self.flist[index]
        
        lq_path = (gt_path.replace('/gt/', '/lq/').replace('\\gt\\', '\\lq\\').replace('_gt.npy', '_lq.npy'))

        # Load raw 1-channel arrays
        gt_img = np.squeeze(array_loader(gt_path)).astype(np.float32)
        lq_img = np.squeeze(array_loader(lq_path)).astype(np.float32)

        gt_img = np.clip(gt_img, 0, None)
        lq_img = np.clip(lq_img, 0, None)

        gt_img = np.clip(np.log1p(gt_img) / self.max_log_val, 0, 1) * 2.0 - 1.0
        lq_img = np.clip(np.log1p(lq_img) / self.max_log_val, 0, 1) * 2.0 - 1.0

        # Convert to float tensor and add the 1-channel dimension: shape [1, H, W]
        gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0)
        lq_tensor = torch.from_numpy(lq_img).float().unsqueeze(0)

        # Resize if images don't match the required U-Net dimensions
        if gt_tensor.shape[1:] != tuple(self.image_size):
            gt_tensor = torch.nn.functional.interpolate(
                gt_tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0)
            lq_tensor = torch.nn.functional.interpolate(
                lq_tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0)

        ret['gt_image'] = gt_tensor
        ret['cond_image'] = lq_tensor
        ret['path'] = gt_path.rsplit("/")[-1].rsplit("\\")[-1]
        return lq_tensor, gt_tensor

    def __len__(self):
        return len(self.flist)

# ==========================================
# 1. 配置 (Ada Lovelace Config)
# ==========================================
class Config:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    img_size = 256
    patch_size = 8
    in_channels = 1
    num_classes = 10
    
    # Modern ViT 架构参数
    hidden_size = 384       
    depth = 8               
    num_heads = 6
    mlp_ratio = 4.0
    bottleneck_dim = 64     
    
    in_context_len = 8      
    in_context_start = 2    
    
    # --- 针对 4060 Ti 的显存优化 ---
    # 4060 Ti 8G 建议: 64
    # 4060 Ti 16G 建议: 128
    batch_size = 64       
    
    # 梯度
    gradient_accumulation_steps = 4
    
    lr = 5e-4
    epochs = 200
    

    num_workers = 0 if platform.system() == 'Windows' else 4
    
    noise_scale = 1.0       
    save_dir = "./jit_results_8"
    resume_path = None

# ==========================================
# 2. 基础组件
# # ==========================================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, num_cls_token=0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len 
        freqs = 1. / (10000 ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len)
        freqs = torch.einsum('i,j->ij', t, freqs) 
        freqs = torch.cat((freqs[:, None, :].repeat(1, pt_seq_len, 1), 
                           freqs[None, :, :].repeat(pt_seq_len, 1, 1)), dim=-1) 
        freqs = rearrange(freqs, 'h w d -> (h w) d') 
        freqs = repeat(freqs, 'n d -> n (d r)', r=2) 
        if num_cls_token > 0:
            pad = torch.zeros(num_cls_token, freqs.shape[1])
            freqs = torch.cat([pad, freqs], dim=0)
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        seq_len = x.shape[-2]
        f_cos = self.freqs_cos[:seq_len]
        f_sin = self.freqs_sin[:seq_len]
        return x * f_cos + x_rotated * f_sin

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    def forward(self, x, c, rope_func):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope_func)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bottleneck_dim):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2).contiguous()

# ==========================================
# 3. JiT 主模型
# ==========================================
class JiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.t_embedder = nn.Sequential(nn.Linear(256, config.hidden_size), nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size))
        self.y_embedder = nn.Embedding(config.num_classes + 1, config.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(config.img_size, config.patch_size, config.in_channels, config.hidden_size, config.bottleneck_dim)
        hw_seq_len = config.img_size // config.patch_size
        head_dim = config.hidden_size // config.num_heads
        self.feat_rope = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=hw_seq_len, num_cls_token=config.in_context_len)
        self.blocks = nn.ModuleList([JiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio) for _ in range(config.depth)])
        self.norm_final = RMSNorm(config.hidden_size)
        self.linear_final = nn.Linear(config.hidden_size, config.patch_size**2 * self.out_channels)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 2 * config.hidden_size))

    def get_time_embedding(self, t):
        half_dim = 256 // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.t_embedder(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))

    def unpatchify(self, x):
        p = self.config.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        return torch.einsum('nhwpqc->nchpwq', x.reshape(shape=(x.shape[0], h, w, p, p, c))).reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        c = self.get_time_embedding(t) + self.y_embedder(y)
        x = self.x_embedder(x)
        for i, block in enumerate(self.blocks):
            if i == self.config.in_context_start:
                ctx = self.y_embedder(y).unsqueeze(1).repeat(1, self.config.in_context_len, 1)
                x = torch.cat([ctx, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.config.in_context_start else self.feat_rope_incontext)
        if self.config.in_context_len > 0: x = x[:, self.config.in_context_len:]
        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        return self.unpatchify(self.linear_final(modulate(self.norm_final(x), shift, scale)))

# ==========================================
# 4. 扩散逻辑 (Flow Matching)
# ==========================================
class Diffusion:
    def __init__(self, config):
        self.device = config.device
        # noise_scale is removed because we use actual noisy X-ray images, not synthetic noise

    def sample_t(self, n):
        # Uniform sampling across [0, 1] is standard for Rectified Flow / paired matching
        return torch.rand(n, device=self.device)

    def q_sample(self, x_clean, x_noisy, t):
        """
        Forward process during training.
        x_clean: Ground truth clean image (x1)
        x_noisy: Input noisy speckle image (x0)
        """
        t = t.view(-1, 1, 1, 1)
        
        # Interpolate between noisy (t=0) and clean (t=1)
        z = t * x_clean + (1 - t) * x_noisy
        
        # Target velocity points exactly from noisy to clean
        target_velocity = x_clean - x_noisy
        
        return z, target_velocity

    @torch.no_grad()
    def p_sample_loop(self, model, noisy_imgs, steps=50):
        model.eval()
        z = noisy_imgs.clone().to(self.device)
        n = z.shape[0]
        dt = 1.0 / steps
        
        for i in range(steps):
            t_curr = torch.tensor(i / steps, device=self.device).repeat(n)
            
            # 1. Generate the dummy label tensor for the sampling batch
            y_dummy = torch.zeros(n, dtype=torch.long, device=self.device)
            
            # 2. Pass z, t_curr, AND y_dummy into the model
            v_pred = model(z, t_curr, y_dummy)
            
            z = z + v_pred * dt
            
        model.train()
        return z.clamp(-1, 1)

# ==========================================
# 5. 主程序 
# ==========================================
def main():
    os.makedirs(Config.save_dir, exist_ok=True)
    print(f"🔥 RTX 4060 Ti Optimizations | Device: {Config.device}")
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = DenoisingDataset(data_root="/Users/nykolasrekasius/xray/xray_speckles/dataset_npy/train", image_size=[256, 256])
    

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
            
    model = JiT(Config).to(Config.device)
    
    # # Windows需先安装triton
    # print("正在编译模型 (torch.compile)...")
    # try:
    #     # mode='reduce-overhead' 会消耗更多显存，这里用默认模式或 max-autotune-no-cudagraphs
    #     model = torch.compile(model) 
    # except Exception as e:
    #     print(f"⚠️ 编译失败或不支持，将使用常规模式运行: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    diffusion = Diffusion(Config)
    # scaler = torch.amp.GradScaler('cuda')
    
    print("🚀 开始训练...")
    global_step = 0
    
    for epoch in range(Config.epochs):
        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        
        optimizer.zero_grad()
        for step, (noisy_imgs, clean_imgs) in enumerate(pbar):
            # 1. Move both noisy and clean paired images to the MPS device
            noisy_imgs = noisy_imgs.to(Config.device, non_blocking=True)
            clean_imgs = clean_imgs.to(Config.device, non_blocking=True)
            
            # 2. Sample random timesteps
            t = diffusion.sample_t(noisy_imgs.shape[0])
            
            # 3. Call the updated q_sample with BOTH images (clean is x1, noisy is x0)
            # Notice we only unpack two variables here now: z and v_target
            z, v_target = diffusion.q_sample(clean_imgs, noisy_imgs, t)

            # Forward

            y_dummy = torch.zeros(z.shape[0], dtype=torch.long, device=Config.device)

            # Remove 'y' (labels) and assign the output directly to v_pred
            v_pred = model(z, t.squeeze(), y_dummy) 
            
            # Calculate loss against the target velocity (which is x_clean - x_noisy)
            loss = torch.nn.functional.mse_loss(v_pred, v_target)
            loss = loss / Config.gradient_accumulation_steps
            
            # Backward
            loss.backward()
            
            # 梯度累积 (Gradient Accumulation)
            if (step + 1) % Config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            # 还原 Loss 用于显示
            current_loss = loss.item() * Config.gradient_accumulation_steps
            if loss_ema is None: loss_ema = current_loss
            else: loss_ema = 0.95 * loss_ema + 0.05 * current_loss
            pbar.set_description(f"E{epoch} | L:{loss_ema:.4f}")

        # 采样测试
        if epoch % 5 == 0:
            print("🎨 正在生成采样图...")
            
            # 1. Grab a real batch of noisy images and ground truth from the dataloader
            test_noisy, test_clean = next(iter(dataloader))
            test_noisy = test_noisy.to(Config.device)
            test_clean = test_clean.to(Config.device)
            
            # 2. Denoise the real images
            with torch.no_grad():
                imgs_denoised = diffusion.p_sample_loop(model, test_noisy)
            
            # 3. Concatenate noisy input, denoised output, and clean ground truth 
            # This stacks them so row 1 is noisy, row 2 is denoised, row 3 is clean
            comparison = torch.cat([test_noisy, imgs_denoised, test_clean], dim=0)
            
            # 4. Save the comparison grid and the model weights
            # nrow=test_noisy.shape[0] ensures each category gets its own row
            save_image((comparison + 1) / 2, f"{Config.save_dir}/{epoch}.png", nrow=test_noisy.shape[0])
            
            torch.save(model.state_dict(), f"{Config.save_dir}/last.pth")

if __name__ == "__main__":
    main()