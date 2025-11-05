# MAE (Masked Autoencoder) Overview 발표 대본

## 1. 기존 비전 학습의 한계 (Introduction)

안녕하세요. 오늘은 **Masked Autoencoder for Computer Vision**, 줄여서 MAE에 대해 소개하겠습니다.

먼저 기존 비전 학습의 한계점을 살펴보겠습니다.

2020년에 발표된 Vision Transformer, ViT는 대용량 데이터와 사전학습을 통해 뛰어난 성능을 보여줬습니다. 하지만 자기지도학습, 즉 레이블 없이 스스로 학습하는 방식으로는 효율적인 표현을 학습하기 어려웠습니다.

그래서 등장한 것이 contrastive learning 기법들입니다. SimCLR, MoCo, BYOL 같은 방법들은 이미지 쌍 간의 관계를 학습합니다. 하지만 이 방법들에도 문제가 있었습니다:
- positive와 negative 쌍을 설계하는 것이 복잡하고
- augmentation에 대한 의존도가 높으며
- 계산량이 매우 많다는 점입니다.

---

## 2. MAE의 배경과 핵심 아이디어 (Background & Motivation)

그렇다면 MAE는 어떻게 이 문제들을 해결했을까요?

배경을 보면, 2018년에 발표된 BERT가 힌트를 제공합니다. BERT는 입력의 일부 토큰을 masking하고, 이를 복원하는 단순한 자기지도 task만으로도 놀라운 성능을 보여줬습니다.

MAE의 저자들은 여기서 영감을 받았습니다. 
**"이미지에서도 이런 mask reconstruction 방식을 적용할 수 있지 않을까?"**

핵심 목표는 간단합니다:
**이미지의 일부를 가리고, 나머지로부터 원본을 복원하도록 학습하면 좋은 시각 표현을 얻을 수 있지 않을까?**

---

## 3. Masked Image Modeling (MIM) (Core Method)

이제 MAE의 핵심 방법인 Masked Image Modeling을 설명드리겠습니다.

MAE는 다음과 같이 작동합니다:

**첫째**, 입력 이미지를 ViT patch 단위로 나눕니다. 예를 들어 16×16 픽셀씩 나누는 거죠.

**둘째**, 이 중 랜덤하게 **75%를 mask**합니다. BERT가 15% 정도만 masking하는 것과 비교하면 매우 높은 비율입니다.

**셋째**, 나머지 **25%의 patch만 인코더에 입력**합니다.

**마지막으로**, 디코더가 mask된 패치를 픽셀 단위로 복원하도록 학습합니다.

이렇게 높은 masking ratio를 사용하는 이유는, 이미지가 텍스트와 달리 중복 정보가 많기 때문입니다. 대부분을 가려야 모델이 진정으로 의미 있는 표현을 학습할 수 있습니다.

---

## 4. 비대칭 구조 (Asymmetric Architecture)

MAE의 또 다른 핵심은 **비대칭 Encoder-Decoder 구조**입니다.

**Encoder 부분**을 보면:
- 입력의 일부, 정확히는 25%만 처리합니다
- 이로 인해 계산량이 크게 절감됩니다
- ViT 구조를 사용합니다 - Patch Embedding과 Transformer Layers로 구성되어 있죠

**Decoder 부분**을 보면:
- mask된 토큰을 포함한 전체 토큰을 입력받습니다
- 하지만 Encoder보다 훨씬 얕고 가볍습니다 - 적은 수의 Transformer block만 사용합니다
- 역할은 원본 이미지를 복원하는 것입니다

**핵심 포인트는 이겁니다**: 
Encoder는 representation 학습에 집중하고, Decoder는 복원 전용입니다. 실제로 사전학습 후에는 Decoder를 버리고 Encoder만 사용합니다.

이 그림을 보시면 구조가 명확히 보입니다.
- 왼쪽: 원본 이미지를 patch로 나누고 대부분을 mask
- 중간: 보이는 patch만 Encoder를 통과
- 오른쪽: Decoder가 전체 이미지를 복원

---

## 5. Reconstruction Target (학습 목표)

그럼 무엇을 복원하도록 학습할까요?

MAE는 입력 이미지를 **patch 단위로 normalize된 pixel value**로 복원합니다.

이 방식의 장점은:
- 색상 정보보다는 구조적 패턴에 집중하게 됩니다
- 단순하지만 효과적입니다

참고로, 이후 연구들에서는 pixel이 아닌 feature-level reconstruction도 시도되었습니다. 예를 들어 iBOT, BEiT 같은 모델들이 있죠.

---

## 6. Pretraining과 Finetuning (전체 파이프라인)

마지막으로 전체 파이프라인을 정리하겠습니다.

**1단계: Pretrain**
- Masked reconstruction task로 대규모 데이터에서 학습합니다
- 예를 들어 ImageNet-1K를 사용합니다
- 이 단계에서 Encoder가 강력한 visual representation을 학습합니다

**2단계: Finetune**
- Encoder만 남기고 Decoder는 버립니다
- classification head를 붙여서 downstream task에 학습시킵니다
- 이미지 분류, 객체 탐지, 세그멘테이션 등 다양한 태스크에 적용 가능합니다

---

## 7. 결론 및 의의 (Conclusion)

MAE는 매우 간단한 아이디어로 큰 성과를 냈습니다:

**장점:**
1. **단순함** - mask하고 복원하는 것만으로도 충분
2. **효율성** - 25%만 encoding하므로 계산량 3배 절감
3. **확장성** - 대규모 데이터에 쉽게 스케일업 가능
4. **범용성** - 다양한 downstream task에 적용 가능

**영향:**
- NLP의 BERT를 비전으로 성공적으로 가져온 첫 사례
- 이후 MAE v2, SimMIM, BEiT 등 많은 후속 연구에 영감을 줌
- Self-supervised learning의 새로운 패러다임 제시

질문 있으시면 말씀해주세요. 감사합니다.

---

## 발표 팁

- **전체 발표 시간**: 약 5-7분
- **각 섹션 시간 배분**:
  - 도입 (1분)
  - 배경 및 동기 (1분)
  - MIM 핵심 방법 (1-2분)
  - 비대칭 구조 (1-2분)
  - Reconstruction Target (30초)
  - 파이프라인 (1분)
  - 결론 (1분)
  
- **그림 활용**: 섹션 4의 MAE 아키텍처 그림을 중심으로 설명
- **강조 포인트**: 75% masking ratio, 비대칭 구조, 효율성
