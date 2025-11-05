# MAE (Masked Autoencoder) 구현 발표 대본

---

## 📌 도입부

안녕하세요. 오늘은 **Masked Autoencoder (MAE)** 를 PyTorch로 처음부터 구현한 코드를 발표하겠습니다.

MAE는 2021년 Meta AI에서 발표한 Self-Supervised Learning 방법으로, 이미지의 일부를 가리고 복원하는 방식으로 학습합니다. BERT의 masking 아이디어를 비전에 적용한 것인데, 놀랍게도 75%의 패치를 가려도 효과적으로 학습이 가능합니다.

오늘 발표는 크게 8개 섹션으로 구성됩니다:
1. Setup & Imports
2. Configuration
3. Utility Functions
4. ViT Token Extractor
5. MAE Model (핵심)
6. Dataset & DataLoader
7. Training/Validation/Test
8. Complete Pipeline

---

## 1️⃣ Setup & Imports

```python
import torch
import torch.nn as nn
import timm  # ViT backbone
from torchvision import transforms, datasets
```

먼저 필요한 라이브러리를 import합니다. 
- **PyTorch**: 기본 딥러닝 프레임워크
- **timm**: Vision Transformer(ViT) 백본을 쉽게 가져오기 위해 사용
- **torchvision**: 데이터 로딩과 전처리용

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

재현성을 위해 모든 랜덤 시드를 고정합니다. GPU를 사용할 경우 CUDA 시드도 함께 설정합니다.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

GPU가 있으면 cuda를, 없으면 cpu를 사용하도록 설정합니다.

---

## 2️⃣ Configuration

```python
@dataclass
class CFG:
    img_size: int = 224
    patch_size: int = 16
    mask_ratio: float = 0.75
```

설정값들을 dataclass로 관리합니다.
- **img_size**: 입력 이미지 크기 (224x224)
- **patch_size**: 한 패치의 크기 (16x16) → 총 196개의 패치가 생성됩니다
- **mask_ratio**: 마스킹 비율 (0.75 = 75%) → MAE의 핵심 하이퍼파라미터

```python
enc_name: str = "vit_base_patch16_224"
dec_dim: int = 384
dec_depth: int = 6
```

- **enc_name**: timm에서 제공하는 ViT Base 모델을 인코더로 사용
- **dec_dim**: 디코더의 hidden dimension (384) - 인코더(768)보다 작습니다
- **dec_depth**: 디코더의 Transformer 레이어 수 (6개) - 인코더(12개)보다 적습니다

이것이 MAE의 **Asymmetric Encoder-Decoder** 구조입니다. 인코더는 크고 깊게, 디코더는 가볍게 만들어서 효율성을 높입니다.

---

## 3️⃣ Utility Functions

### Patchify - 이미지를 패치로 분해

```python
class Patchify(nn.Module):
    def forward(self, imgs):  # (B,C,H,W) -> (B, N, P*P*C)
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, h*w, p*p*C)
```

입력 이미지 (B, 3, 224, 224)를 패치 시퀀스 (B, 196, 768)로 변환합니다.
- 224÷16 = 14, 따라서 14×14 = 196개의 패치
- 각 패치는 16×16×3 = 768 차원의 벡터

### Unpatchify - 패치를 다시 이미지로 복원

```python
class Unpatchify(nn.Module):
    def forward(self, x):  # (B,N,P*P*C) -> (B,C,H,W)
```

Patchify의 역연산입니다. 디코더가 복원한 패치들을 다시 이미지 형태로 재구성합니다.

### random_mask_indices - 랜덤 마스킹

```python
def random_mask_indices(num_patches, mask_ratio=0.75):
    n_mask = int(num_patches * mask_ratio)
    ids = torch.randperm(num_patches)
    mask_ids = ids[:n_mask]  # 가릴 패치
    keep_ids = ids[n_mask:]  # 보여줄 패치
    return keep_ids, mask_ids
```

196개 패치 중 75%(147개)는 mask_ids, 25%(49개)는 keep_ids로 분류합니다.
- **keep_ids**: 인코더에 입력으로 들어갈 패치 (보이는 부분)
- **mask_ids**: 디코더가 복원해야 할 패치 (가려진 부분)

---

## 4️⃣ ViT Token Extractor

```python
def vit_tokens_from_timm(vit: nn.Module, imgs: torch.Tensor):
    x = vit.patch_embed(imgs)  # (B, N, D)
```

timm의 ViT 모델에서 토큰 시퀀스를 추출하는 헬퍼 함수입니다.

```python
cls_token = vit.cls_token.expand(B, -1, -1)  # (B,1,D)
x = torch.cat((cls_token, x), dim=1)         # (B, N+1, D)
```

**CLS 토큰**을 추가합니다. 이미지 전체의 요약 정보를 학습하는 특수 토큰입니다.

```python
x = x + vit.pos_embed
```

**Position Embedding**을 추가합니다. Transformer는 순서 정보가 없기 때문에, 각 토큰이 이미지의 어느 위치 패치인지 알려줘야 합니다.

```python
for blk in vit.blocks:
    x = blk(x)
x = vit.norm(x)
return x  # (B, N+1, D)
```

ViT의 Transformer 블록들을 통과시킨 후, Layer Normalization을 적용하고 반환합니다.

---

## 5️⃣ MAE Model - 핵심 구현

자, 이제 가장 중요한 MAE 모델 구현입니다.

### __init__ - 모델 구조 정의

```python
class MAE(nn.Module):
    def __init__(self, cfg: CFG):
        # Encoder (timm ViT)
        self.encoder = timm.create_model(cfg.enc_name, pretrained=False)
        emb_dim = self.encoder.embed_dim  # 768
```

**인코더**는 timm의 ViT Base를 그대로 사용합니다. 마스크되지 않은 패치들만 입력으로 받습니다.

```python
        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
```

**mask_token**: 마스크된 위치를 대체할 학습 가능한 토큰입니다. 디코더는 이 토큰을 보고 원래 패치를 복원해야 합니다.

```python
        self.dec_pos = nn.Parameter(torch.zeros(1, total_tokens, cfg.dec_dim))
```

디코더용 **Position Embedding**입니다. 인코더와 별도로 학습됩니다.

```python
        self.enc_to_dec = nn.Linear(emb_dim, cfg.dec_dim)
```

인코더의 출력(768차원)을 디코더의 입력(384차원)으로 변환하는 projection layer입니다.

```python
        layer = nn.TransformerEncoderLayer(d_model=cfg.dec_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=cfg.dec_depth)
```

**디코더**는 가벼운 Transformer입니다. 6개 레이어, 384 차원으로 인코더(12레이어, 768차원)보다 훨씬 작습니다.

```python
        self.head = nn.Linear(cfg.dec_dim, cfg.patch_size * cfg.patch_size * 3)
```

최종 출력 head입니다. 384차원을 16×16×3 = 768 픽셀값으로 변환합니다.

### forward - 순전파 로직

```python
    def forward(self, imgs: torch.Tensor):
        # 1) target patches
        target = self.patchify(imgs)  # (B, N, P2*C)
```

먼저 입력 이미지를 패치로 나눕니다. 이것이 우리가 복원해야 할 **정답(target)**입니다.

```python
        # 2) mask indices per-sample
        keep_ids, mask_ids = [], []
        for _ in range(B):
            k, m = random_mask_indices(N, self.cfg.mask_ratio)
            keep_ids.append(k); mask_ids.append(m)
```

배치의 각 샘플마다 독립적으로 랜덤 마스킹을 수행합니다.
- **keep_ids**: 인코더에 넣을 패치 (보이는 25%)
- **mask_ids**: 가릴 패치 (숨길 75%)

```python
        # 3) Encoder tokens
        enc_all = vit_tokens_from_timm(self.encoder, imgs)   # (B, N+1, De)
        enc_tokens = enc_all[:, 1:, :]                       # (B, N, De)
```

**중요한 점**: 인코더는 전체 이미지를 봅니다! 
실제로는 keep 위치만 사용하지만, ViT 구조상 전체 이미지를 넣어 패치 임베딩을 만든 후 필요한 부분만 선택합니다.

CLS 토큰(첫 번째 토큰)은 제외하고 패치 토큰만 사용합니다.

```python
        enc_kept = torch.gather(
            enc_tokens, dim=1,
            index=keep_ids.unsqueeze(-1).expand(-1, -1, enc_tokens.size(-1))
        )  # (B, Nk, De)
```

`torch.gather`로 keep_ids에 해당하는 토큰만 선택합니다. 
이것이 **인코더가 실제로 본 토큰들의 latent representation**입니다.

```python
        # 4) Decoder input: kept + mask
        dec_kept = self.enc_to_dec(enc_kept)      # (B, Nk, Dd)
        dec_mask = self.mask_token.expand(B, Nm, -1)
        dec_in = torch.cat([dec_kept, dec_mask], dim=1) + self.dec_pos[:, :Nk+Nm, :]
```

디코더 입력을 구성합니다:
1. **enc_kept**: 인코더가 본 패치들 (Projection 후)
2. **mask_token**: 가려진 패치들을 대체할 학습 가능한 토큰
3. 두 개를 concatenate하고 position embedding 추가

이게 MAE의 핵심 아이디어입니다! 디코더는 일부 실제 정보(kept)와 mask token을 모두 받아서 복원합니다.

```python
        dec_out = self.decoder(dec_in)            # (B, Nk+Nm, Dd)
        pred = self.head(dec_out[:, Nk:, :])      # (B, Nm, P2*C)
```

디코더를 통과시킨 후, **마스크된 부분만** 예측합니다 (`[:, Nk:]`).
kept 부분은 이미 정답을 아니까 loss 계산에서 제외합니다.

```python
        target_masked = torch.gather(
            target, dim=1,
            index=mask_ids.unsqueeze(-1).expand(-1, -1, target.size(-1))
        )
        loss = F.mse_loss(pred, target_masked)
```

마스크된 패치의 정답(target_masked)과 예측(pred)의 MSE loss를 계산합니다.

**픽셀 레벨 복원**을 수행하는 것입니다. 각 패치의 768개 픽셀값을 정확히 맞추도록 학습합니다.

---

## 6️⃣ Dataset & DataLoader

```python
def build_dataloaders(cfg: CFG):
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor()
    ])
```

간단한 전처리만 수행합니다:
- 224×224로 resize
- Tensor로 변환 (0~1 정규화 자동 적용)

**중요**: MAE는 self-supervised이므로 레이블이 필요 없습니다! 이미지 자체가 입력이자 정답입니다.

```python
    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(str(train_dir), transform=tfm)
    else:
        train_ds = FakeData(size=256, ...)
```

실제 데이터가 있으면 ImageFolder로 로드하고, 없으면 FakeData로 데모를 수행합니다.

---

## 7️⃣ Training / Validation / Test

### train_one_epoch

```python
def train_one_epoch(model, dl, opt, epoch, cfg: CFG):
    model.train()
    for imgs, _ in dl:
        loss, pred, idx, target = model(imgs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
```

전형적인 PyTorch 학습 루프입니다.
- Forward pass로 loss 계산
- Backward pass로 gradient 계산
- Optimizer로 파라미터 업데이트

레이블(\_)은 사용하지 않습니다. Self-supervised learning의 특징입니다.

### validate

```python
@torch.no_grad()
def validate(model, dl, cfg: CFG, save_samples=False):
    model.eval()
    # ... validation loss 계산
    if save_samples:
        save_grid(imgs[:16].cpu(), f"{cfg.save_dir}/viz/input_epoch.jpg")
```

Validation 중에 입력 이미지를 저장해서 나중에 복원 결과와 비교할 수 있습니다.

---

## 8️⃣ Complete Pipeline

```python
model = MAE(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
```

MAE 논문에서 권장하는 설정:
- **AdamW** optimizer (weight decay 포함)
- **Cosine annealing** learning rate scheduler
- Learning rate: 1e-4, Weight decay: 0.05

```python
for epoch in range(1, cfg.epochs+1):
    tr = train_one_epoch(model, train_dl, opt, epoch, cfg)
    va = validate(model, val_dl, cfg, save_samples=(epoch % 1 == 0))
    sch.step()
```

전체 학습 루프입니다:
1. 한 에폭 학습
2. Validation 수행
3. Learning rate 조정
4. 베스트 모델 저장

---

## 🎯 핵심 요약

MAE의 핵심 아이디어 3가지:

### 1. High Masking Ratio (75%)
- BERT는 15% 마스킹 vs MAE는 75% 마스킹
- 비전은 redundancy가 높아서 가능
- 계산 효율도 3배 향상 (인코더가 25%만 처리)

### 2. Asymmetric Encoder-Decoder
- **Encoder**: 크고 깊게 (ViT-Base, 768dim, 12 layers)
  - 보이는 25% 패치로 강력한 representation 학습
- **Decoder**: 작고 가볍게 (384dim, 6 layers)
  - 복원은 상대적으로 쉬운 작업
  - Pre-training 후 버려짐 (downstream task는 encoder만 사용)

### 3. Pixel-level Reconstruction
- Normalized pixel values를 직접 예측
- MSE loss로 간단하게 학습
- 대조 학습(contrastive)보다 구현이 쉽고 효과적

---

## 🚀 실제 사용법

```python
# 1) Pre-training (이 코드)
model = MAE(cfg).to(device)
# ... 대용량 unlabeled 데이터로 학습

# 2) Fine-tuning (downstream task)
encoder = model.encoder  # 학습된 인코더만 추출
classifier = nn.Linear(768, num_classes)  # Classification head 추가
# ... labeled 데이터로 fine-tuning
```

MAE로 pre-training한 인코더는 다양한 downstream task에 사용할 수 있습니다:
- Image Classification
- Object Detection
- Semantic Segmentation
- 등등

---

## 📊 MAE의 장점

1. **데이터 효율성**: 레이블 없이 학습 가능
2. **계산 효율성**: 25% 패치만 인코딩 → 3배 빠름
3. **확장성**: 모델 크기를 키울수록 성능 향상
4. **범용성**: 다양한 downstream task에 전이 가능
5. **구현 간단성**: 대조 학습보다 구조가 단순

---

## 🎬 마무리

오늘 발표에서는 MAE의 전체 구현을 단계별로 살펴봤습니다:

1. ✅ 이미지를 패치로 나누기 (Patchify)
2. ✅ 랜덤하게 75% 마스킹
3. ✅ 보이는 25%만 인코더에 입력
4. ✅ 디코더로 가려진 75% 복원
5. ✅ MSE loss로 학습

핵심은 **"적게 보고, 많이 복원하기"** 입니다.

이 코드를 실행하면 실제로 MAE를 학습시키고, 마스크된 이미지를 복원하는 것을 확인할 수 있습니다.

질문 있으시면 편하게 해주세요. 감사합니다! 🙏

---

## 📚 참고자료

- **논문**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., CVPR 2022)
- **GitHub**: https://github.com/facebookresearch/mae
- **timm library**: https://github.com/huggingface/pytorch-image-models
