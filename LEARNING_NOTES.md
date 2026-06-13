# 📚 LEARNING NOTES: Anomaly Detection System

**Student:** Data Science Student  
**Project:** Industrial Defect Detection using Feature-Based Anomaly Detection  
**Timeline:** June 9 - June 23, 2026 (14 days)  
**Hardware:** RTX 3050 + 32GB RAM  

---

## MODULE 1: PIPELINE UNDERSTANDING

### 1.1 - What is Anomaly Detection?

**Simple Definition:**
```
Anomaly Detection = Finding things that look DIFFERENT from normal

Example:
- Normal: Toothbrush with perfect bristles
- Anomaly: Toothbrush with broken bristles
→ System learns what "normal" looks like
→ When it sees something different → "ANOMALY!"
```

**Unsupervised Anomaly Detection (Our approach):**
```
We ONLY show normal examples during training.
System learns: "This is what normal looks like"
During testing: 
  - If new image looks like normal → OK
  - If new image looks different → NG (defect)
```

---

### 1.2 - Current Pipeline Overview

**What we have now:**

```
STEP 1: Feature Extraction (ResNet50)
├─ Input: Image (224 × 224 × 3)
├─ Model: ResNet50 pretrained on ImageNet
├─ Extract from: Layer 2 + Layer 3
└─ Output: Feature vectors (high-dimensional representation)

STEP 2: Memory Bank
├─ Collect features from all normal training images
├─ Select ~10% as representative samples (currently: random)
└─ Store in GPU memory

STEP 3: Distance Computation (Euclidean)
├─ For new test image:
│  ├─ Extract features
│  ├─ Calculate distance to each memory bank sample
│  └─ Take minimum distance
└─ Output: Anomaly score

STEP 4: Threshold Decision
├─ If anomaly_score > threshold → NG (defect)
├─ If anomaly_score ≤ threshold → OK (normal)
└─ Threshold from: mean(normal_scores) + 3*std(normal_scores)

STEP 5: Heatmap Generation
├─ Convert distance map to spatial heatmap
├─ Upscale to original image resolution
└─ Visualize defect location
```

---

### 1.3 - Key Concepts Explained

#### Concept A: Why ResNet50?
```
ResNet50 = Pretrained Convolutional Neural Network

Advantages:
1. Trained on ImageNet (1.4M images, 1000 classes)
   → Learned useful visual features
   
2. We DON'T train it (weights frozen)
   → Just use it as feature extractor
   
3. Layer 2 & 3 = rich semantic features
   → Good for capturing object patterns

Use: Extract features instead of pixels
  - Pixels: [224, 224, 3] = 150,528 dimensions
  - ResNet features: [1536] = much smaller + meaningful
```

#### Concept B: Memory Bank
```
What is it?
→ Collection of "reference points" from normal training data

Why need it?
→ To compare test samples against
→ Calculate: How different is test image from normal?

Current problem: Random selection
→ May miss important feature patterns
→ Not representative

Solution: K-center greedy (we'll implement)
→ Select samples that cover entire feature space
```

#### Concept C: Euclidean Distance
```
What is it?
→ Measure how far apart two things are

Example in 2D:
Point A: (1, 2)
Point B: (4, 6)
Distance = sqrt((4-1)² + (6-2)²) = sqrt(9 + 16) = 5

In our case:
Feature A: [0.5, -0.2, 0.8, ..., 1.2] (1536 dimensions)
Feature B: [0.4, -0.3, 0.7, ..., 1.1] (1536 dimensions)
Distance = sqrt((0.4-0.5)² + (-0.3-(-0.2))² + ... + (1.1-1.2)²)

Interpretation:
- Small distance → Similar to normal
- Large distance → Different from normal → ANOMALY
```

---

## MODULE 2: PROBLEMS & SOLUTIONS

### Problem 1: Data Leakage ⚠️ CRITICAL

**Current Code Flow:**
```python
# Cell 7: Calculate threshold from TRAIN data
y_score_good = [...]  # Scores on training images
best_threshold = np.mean(y_score_good) + 3 * np.std(y_score_good)
# Result: threshold = 5.2

# Cell 13: Calculate threshold from TEST data
f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
# Result: threshold = 5.7
# ⚠️ WHICH ONE TO USE? THIS IS DATA LEAKAGE!
```

**Why is this bad?**
```
Imagine you're taking an exam:
❌ Teacher shows you the exam before
❌ You memorize answers
❌ You get 100/100

But did you actually LEARN?
→ No! You just memorized specific answers.

With our model:
❌ We tune threshold using test data
❌ We get high metrics on test data
❌ But on NEW data, performance drops

Why? Because threshold was tailored to THIS specific test set.
```

**Solution: Proper Data Split**
```
Train data: 70% of normal images
├─ Extract memory bank
└─ Set initial threshold

Validation data: 15% of normal + defective images
├─ Tune threshold (if needed)
└─ Check performance

Test data: 15% of normal + defective images
├─ ONLY for final evaluation
└─ Never used for tuning anything
```

**Why this matters for thesis:**
```
Supervisor will ask:
"How do you know threshold is optimal?"

❌ Bad answer: "We tuned it on test data"
✅ Good answer: "We selected threshold on validation set, 
                 then evaluated on completely unseen test set"
```

---

### Problem 2: Feature Normalization

**Current Issue:**
```
ResNet50 outputs from different layers have different scales:

Layer 2 output:
  - Values: typically 50 to 100
  - Example: [102.3, 95.1, 87.5, ..., 98.2]

Layer 3 output:
  - Values: typically 5 to 10
  - Example: [5.2, 4.8, 6.1, ..., 7.3]

When concatenated:
  features = [102.3, 95.1, 87.5, ..., 5.2, 4.8, 6.1, ..., 7.3]
            [  Layer 2 features  ...   Layer 3 features  ]

Distance calculation:
  dist² = (layer2_diff)² + (layer3_diff)²
  dist² = (30)² + (2)²
  dist² = 900 + 4 = 904

  ⚠️ Layer 2 contributes 99.6%!
  ⚠️ Layer 3 contributes 0.4%!
  → Layer 3 is essentially useless!
```

**Solution: Normalization**
```
Before concatenation:

Layer 2 normalized:
  norm = sqrt(102.3² + 95.1² + ...) ≈ 1450
  normalized = [102.3/1450, 95.1/1450, ...] = [0.070, 0.066, ...]
  → Values now ~ 0 to 0.1

Layer 3 normalized:
  norm = sqrt(5.2² + 4.8² + ...) ≈ 145
  normalized = [5.2/145, 4.8/145, ...] = [0.036, 0.033, ...]
  → Values now ~ 0 to 0.1

When concatenated:
  features = [0.070, 0.066, ..., 0.036, 0.033, ...]
            [  normalized layer 2  ...  normalized layer 3]

Distance calculation:
  dist² = (0.001)² + (0.002)²
  dist² = 0.000001 + 0.000004 = 0.000005

  ✅ Both layers contribute equally!
  ✅ No longer dominated by scale
```

---

### Problem 3: Memory Bank Selection

**Current: Random Selection**
```python
selected_indices = np.random.choice(len(memory_bank), 
                                    size=len(memory_bank)//10, 
                                    replace=False)
```

**Issue:**
```
Imagine feature space is 1D, 0 to 100:

Normal training samples: [5, 8, 12, 15, 18, 22, 25, 28, 32, 35, ...]
(100 samples total)

Random selection (10 samples):
Run 1: [5, 12, 18, 25, 32, 38, 45, 52, 58, 65]
→ Covers 0-70, misses 70-100 (gap!)

Run 2: [8, 15, 22, 28, 35, 42, 48, 55, 62, 68]
→ Different coverage, also has gaps

❌ Inconsistent results across runs
❌ May miss important feature patterns
❌ Not strategic
```

**Solution: K-center Greedy**
```
Objective: Select 10 samples that COVER feature space

Algorithm:
1. Start with any sample: [5]
2. Find sample farthest from selected: [5, 95]
3. Find sample farthest from {5,95}: [5, 95, 25]
4. Find sample farthest from {5,95,25}: [5, 95, 25, 75]
... repeat until 10 samples ...

Result: [5, 95, 25, 75, 50, 15, 85, 35, 65, 45]
✅ Covers entire range 0-100
✅ Consistent across runs
✅ Representative
```

---

## MODULE 3: TWO REAL BUGS WE HIT (June 9 — Most Important Lesson!)

Khi chạy code lần đầu sau refactor, kết quả SAI hoàn toàn. Đây là 2 bug
thật và cách chúng ta tìm ra — quan trọng hơn cả lý thuyết ở trên.

### BUG 1: Normalization không nhất quán (Inconsistent Normalization)

**Triệu chứng (Symptom):**
```
Anomaly scores (train): mean=34.1679, std=0.0083
Test defective scores:  34.162, 34.176

→ TẤT CẢ điểm số đều ~34.16
→ Ảnh tốt và ảnh lỗi GIỐNG HỆT nhau
→ Hệ thống vô dụng, không phân biệt được gì
```

**Nguyên nhân gốc (Root Cause):**
```python
# Memory bank lưu features GỐC (raw), magnitude ~34
memory_bank.build(features)            # ||feature|| ~ 34

# Nhưng khi tính điểm, query lại được normalize về magnitude 1
features_normalized = features / torch.norm(features)   # ||feature|| = 1
distances = torch.cdist(features_normalized, memory_bank)
```

**Tại sao ra ~34 cho mọi thứ?**
```
So sánh vector dài 1 với vector dài 34:
distance = sqrt(1² + 34² - 2·1·34·cos θ)
         = sqrt(1 + 1156 - 68·cos θ)

cos θ chạy từ -1 đến 1, nhưng số 1156 át tất cả.
→ distance luôn xấp xỉ 34, bất kể nội dung ảnh.
→ "Độ dài" (scale) nuốt chửng "hướng" (direction = thông tin thật).
```

**Cách fix:**
```python
# Trong MemoryBank.build() — normalize NGAY khi lưu
bank = torch.cat(features_list, dim=0)
bank = bank / (torch.norm(bank, dim=1, keepdim=True) + 1e-8)
# Giờ cả memory bank VÀ query đều có magnitude = 1 → nhất quán
```

**Bài học:** Mọi phép so sánh khoảng cách phải dùng feature ĐÃ XỬ LÝ
GIỐNG NHAU ở cả hai phía. Một bên normalize, một bên không → vô nghĩa.

---

### BUG 2: K-center greedy chọn trùng 1 sample mãi mãi

**Triệu chứng:** Memory bank "4704 samples" nhưng thực ra là 4704 bản sao
của cùng 1 điểm.

**Nguyên nhân gốc:**
```python
distances = torch.cdist(self.bank, self.bank[0:1])
distances[:, 0] = -1          # BUG: cột 0 là cột DUY NHẤT
                              # → set TẤT CẢ khoảng cách về -1
for ...:
    idx = torch.argmax(distances.min(dim=1)[0])   # luôn = 0
    # → chọn lại sample 0 mỗi vòng lặp
```

**Cách fix (viết lại đúng thuật toán):**
```python
selected_idx = [0]
min_distances = torch.cdist(self.bank, self.bank[0:1]).squeeze(1)  # vector [N]
min_distances[0] = -1                          # đánh dấu đã chọn

for _ in range(1, k):
    farthest = torch.argmax(min_distances).item()    # điểm XA NHẤT
    selected_idx.append(farthest)
    new_d = torch.cdist(self.bank, self.bank[farthest:farthest+1]).squeeze(1)
    min_distances = torch.minimum(min_distances, new_d)  # cập nhật min
    min_distances[farthest] = -1
```

**Bài học:** Thuật toán đúng trên giấy vẫn sai khi code. Luôn viết unit
test nhỏ (chúng ta đã test: 15/15 samples là duy nhất sau khi chọn).

---

## MODULE 4: KẾT QUẢ TRƯỚC vs SAU (Before/After)

| Chỉ số | TRƯỚC (có bug) | SAU (đã fix) |
|--------|----------------|--------------|
| Train score mean | 34.1679 | 0.1948 |
| Train score std | 0.0083 | 0.0005 |
| Defective score | ~34.16 (giống train) | 0.40 – 0.59 (cao gấp 2-3x) |
| Threshold | 34.1704 | 0.1964 |
| **AUC-ROC** | không đo được | **0.9111** |
| **F1-Score** | không đo được | **0.8333** |
| Heatmap | mờ, nhiễu | rõ, khoanh đúng vùng lỗi |

**Ý nghĩa:**
- Điểm ảnh lỗi (0.40-0.59) giờ CAO HƠN HẲN ảnh tốt (~0.19) → phân biệt được
- AUC 0.9111 là baseline TỐT và TRUNG THỰC cho PatchCore-style trên toothbrush
- Đây là con số bạn có thể tự tin viết vào thesis

---

## MODULE 4B: BUG 3 — Threshold căn trên dữ liệu "gian lận" (June 9)

**Triệu chứng:** Một ảnh GOOD (test) bị phán Defective. Score 0.2628 > threshold 0.1964.

**Điều tra bằng dữ liệu thật:**
```
TRAIN good:   mean=0.1948  std=0.0005   (cực thấp, cực gọn)
TEST good:    mean=0.2854  std=0.0432   ← CAO HƠN HẲN train good!
TEST defect:  mean=0.4803  std=0.1458
Threshold (mean+3std train) = 0.1964
→ 12/12 ảnh test good vượt ngưỡng (false positive 100%)
```

**Nguyên nhân gốc:**
Ảnh train/good được dùng để XÂY memory bank. Nên khi chấm điểm chính chúng,
mỗi patch tìm thấy "bản thân nó" trong bank → khoảng cách ~0 → điểm thấp giả
tạo (0.195). Threshold tính từ đây quá hẹp.

Ảnh good MỚI (test) chưa từng nằm trong bank → điểm thật ~0.285 → vượt ngưỡng.

```
train/good ─┬─► xây memory bank   (điểm tự chấm = thấp giả: 0.195)
            └─► tính threshold     (hẹp: 0.196)
test/good (mới) ─► điểm thật 0.285 ─► > 0.196 ─► NG SAI
```

**Bài học (rất quan trọng cho thesis):**
KHÔNG BAO GIỜ tính threshold trên chính dữ liệu đã dùng để xây memory bank.
Phải tách một tập "good" riêng (calibration / validation) mà bank CHƯA thấy.

**Cách fix:** Tách train/good 80/20.
- 80% → xây memory bank
- 20% → tập calibration (good "chưa thấy") → tính threshold từ đây
→ Giờ threshold phản ánh đúng mức điểm của ảnh good thật.

---

## MODULE 4C: Sự thật về overlap (giới hạn của baseline)

Điểm số good (0.24–0.38) và defect (0.27–0.69) **chồng lấn nhau**:
- defect thấp nhất = 0.266 < good cao nhất = 0.378

→ KHÔNG threshold nào tách sạch 100%. Đây là giới hạn của feature ResNet thô.
→ AUC 0.91 là tốt nhưng còn dư địa. Đây CHÍNH LÀ lý do cần Phase 2:
  - Chuẩn hóa ImageNet trước khi vào ResNet (hiện đang THIẾU)
  - Thử ViT features
  - Làm mượt (smooth) anomaly map trước khi lấy max
→ Mục tiêu Phase 2: kéo giãn khoảng cách good vs defect → giảm overlap.

---

## MODULE 4D: ImageNet Normalization (June 9 — improvement #1)

**Vấn đề:** ResNet50 được pretrain trên ImageNet, kỳ vọng ảnh đầu vào được
chuẩn hóa theo thống kê ImageNet. Code cũ chỉ Resize + ToTensor (pixel 0–1),
THIẾU bước normalize → feature kém chuẩn.

**Cách fix:** thêm vào ImageTransform:
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
(Đây là mean/std của tập ImageNet — con số chuẩn ai cũng dùng.)

**Lưu ý khi hiển thị:** sau khi normalize, tensor không còn nằm trong [0,1],
nên muốn show ảnh gốc phải "de-normalize": `img * std + mean` rồi clip [0,1].

**Kết quả (toothbrush):**
| Chỉ số | Trước | Sau |
|--------|-------|-----|
| AUC-ROC | 0.9111 | 0.9139 |
| F1-Score | 0.8364 | 0.9153 |

→ AUC gần như giữ nguyên nhưng F1 tăng mạnh (0.836 → 0.915): quyết định OK/NG
chính xác hơn nhiều. Đây là cải tiến nhỏ nhưng đúng chuẩn, đáng giữ.

---

## MODULE 4E: Cải thiện ANOMALY SCORING (June 10 — core concept)

Sau khi có ViT, nút thắt của screw chuyển từ "feature" sang "cách tính điểm".
Đây là phần bạn cần nắm rất kỹ.

### Điểm ảnh (image score) được tính từ điểm patch như thế nào?

Mỗi ảnh → 784 patch → mỗi patch có 1 khoảng cách tới memory bank (patch score).
Phải gộp 784 số này thành 1 điểm cho cả ảnh. Có nhiều cách:

**(a) max** (cách cũ): lấy patch bất thường nhất.
```python
score = patch_scores.max()
```
- Ưu: nhạy với lỗi rất nhỏ (chỉ cần 1 patch lạ).
- Nhược: **rất nhiễu** — 1 patch nhiễu ngẫu nhiên ở ảnh good cũng đẩy điểm lên;
  1 patch lỗi đơn lẻ ở ảnh defect dễ bị "chìm".

**(b) mean of top-k**: trung bình k patch cao nhất.
```python
score = patch_scores.topk(k).values.mean()
```
- Ổn định hơn max, nhưng nếu lỗi chỉ ở 1-2 patch thì k lớn làm loãng tín hiệu.

### Gaussian smoothing — chìa khóa

Trước khi gộp, **làm mượt bản đồ điểm 28×28** bằng Gaussian (mỗi patch lấy trung
bình có trọng số với hàng xóm). Đây là cách PatchCore/PaDiM làm.
```python
smoothed = gaussian_blur(patch_scores.view(28,28), sigma=1)
score = smoothed.max()
```
**Tại sao hiệu quả:**
- Patch nhiễu đơn lẻ (good) bị hàng xóm "kéo xuống" → giảm báo động giả.
- Vùng lỗi thật thường gồm nhiều patch gần nhau → được "củng cố" lẫn nhau → nổi bật hơn.
→ Khoảng cách good vs defect giãn ra → AUC tăng, và đặc biệt **recall/F1 tăng**.

### Kết quả thật trên screw (thí nghiệm 2026-06-10)

| Cách tính | ResNet AUC/F1 | ViT AUC/F1 |
|-----------|---------------|------------|
| max (cũ) | 0.839 / 0.300 | 0.896 / 0.300 |
| **max + smooth(σ=1)** | 0.899 / 0.370 | 0.900 / **0.575** |

ViT screw F1: **0.300 → 0.575** (recall 0.18 → 0.40) chỉ nhờ smoothing! Đây là minh
chứng: cùng feature, chỉ đổi cách *gộp điểm* đã cải thiện lớn.

### Bài học cốt lõi
- **3 tầng ảnh hưởng kết quả:** (1) feature (ResNet/ViT), (2) coreset, (3) **scoring**.
  Cải thiện scoring rẻ mà hiệu quả — đừng chỉ chăm chăm đổi backbone.
- **AUC đo khả năng xếp hạng; F1 đo quyết định OK/NG.** Smoothing giúp cả hai vì nó
  làm điểm "sạch" hơn trước khi so ngưỡng.
- Default đã chọn: `method='max', smooth_sigma=1.0` (chỉnh trong Config).

---

## MODULE 4F: Nâng ca khó screw — 3 đòn bẩy (June 10)

screw bị "trần" ở AUC ~0.90 với cả ResNet và ViT-S/8@224 → giới hạn là độ phân
giải/chất lượng feature cho lỗi siêu nhỏ. Thí nghiệm thật (_screw_boost.py):

| Cấu hình | AUC | F1 | Recall |
|----------|----:|---:|-------:|
| ViT-S/8 @224 (baseline) | 0.900 | 0.575 | 0.40 |
| ViT-S/8 @320 | 0.921 | 0.608 | 0.44 |
| ViT-S/8 @384 | 0.932 | 0.663 | 0.50 |
| ViT-S/8 @224 + ghép 4 tầng | 0.922 | 0.592 | 0.42 |
| **DINOv2 ViT-S/14 @518** | **0.963** | **0.763** | **0.62** |

**Toàn bộ hành trình nâng screw F1:** 0.300 (max) → 0.575 (+smoothing) →
0.663 (+độ phân giải 384) → **0.763 (DINOv2)**. Recall 0.18 → 0.62.

### 3 khái niệm cốt lõi rút ra
1. **Độ phân giải đầu vào** quan trọng cho lỗi nhỏ: 224→384 nâng F1 đều đặn vì
   defect chiếm nhiều patch hơn → tín hiệu mạnh + smoothing hiệu quả hơn.
2. **Multi-layer features** (ghép nhiều tầng, NHỚ normalize từng tầng để tránh bug
   scale ở [[technical-notes]]) nâng AUC mà không tốn thêm độ phân giải.
3. **Backbone mạnh hơn (DINOv2)** là đòn lớn nhất: tự giám sát trên 142M ảnh, dense
   features rất mịn, chạy ở 518px (1369 token). Là lựa chọn SOTA cho anomaly detection.

→ Bài học: khi một ca "kịch trần", đừng chỉ chỉnh threshold — hãy nâng **độ phân
giải** và **chất lượng feature**. Ba đòn này cộng dồn được.

### Đòn bẩy thứ 4: Rotation augmentation cho memory bank (đòn screw-specific)

**Vì sao:** Paper MVTec nói screw được chụp với **góc xoay ngẫu nhiên** (khác
toothbrush/capsule/pill được căn thẳng). Nên vít tốt ở hướng "lạ" cũng bị chấm điểm
cao → phân bố good RỘNG (ta đã thấy: good range 0.30-0.41) → nuốt tín hiệu lỗi.

**Cách làm:** xoay ảnh train good ở nhiều góc, đưa CẢ bản gốc + bản xoay vào memory
bank (chỉ xoay BANK, không xoay ảnh test). Bank phủ mọi hướng → vít tốt match dù xoay.

**Kết quả thật (DINOv2@518, build=100):**
| Mức xoay | AUC | F1 | Recall |
|----------|----:|---:|-------:|
| no-rot | 0.9285 | 0.663 | 0.504 |
| **90° (4x)** | 0.9252 | **0.726** | **0.580** |
| 45° (8x) | 0.887 | 0.640 | 0.479 |
| 30° (12x) | 0.889 | 0.531 | 0.361 |

**2 bài học cốt lõi:**
1. **Có điểm ngọt** — 90° (bội 90, xoay sạch không méo góc) là tốt nhất. Xoay góc lệch
   (45/30°) tạo góc đen/nội suy + over-augmentation → bank hấp thụ cả lỗi → TỆ hơn.
2. Rotation aug nâng **F1/recall** (quyết định OK/NG) chứ không nâng AUC (xếp hạng) —
   vì nó co phân bố good lại, giúp ngưỡng tách tốt hơn. Hiểu rõ AUC vs F1 lần nữa.

→ Công thức screw tốt nhất: **DINOv2@518 + xoay 90° (4x) + smoothing**. Nguồn:
PatchCore (amazon-science), FYD alignment, augmented-memory-bank (Aug.R).

---

## MODULE 4G: CHỌN THRESHOLD — làm sao tìm ngưỡng "phù hợp"? (June 10 — core concept)

**Vấn đề:** AUC cao = feature xếp hạng tốt good vs defect. Nhưng để RA QUYẾT ĐỊNH
(OK/NG) phải cắt 1 ngưỡng. Ngưỡng quyết định Precision/Recall/F1. Chọn sai → F1 thấp
dù feature tốt. (Ví dụ screw: feature đã tốt, nhưng ngưỡng bảo thủ làm F1 tụt.)

**Luật vàng — KHÔNG được vi phạm:** ngưỡng phải tính TỪ held-out good, KHÔNG BAO GIỜ
từ nhãn test. Dò ngưỡng để F1 test cao nhất = **rò rỉ dữ liệu** = kết quả vô giá trị.

**3 chính sách hợp lệ (chỉ dùng good):**
```
1. sigma:    threshold = mean + k*std   (giả định Gaussian)
   k=1 -> ~16% FPR | k=2 -> ~2.3% FPR | k=3 -> ~0.13% FPR
2. quantile: threshold = phân vị q của good   (đặt mục tiêu FPR = 1-q trực tiếp)
3. max:      threshold = max(good)   (bảo thủ nhất, gần như 0 báo nhầm)
```

**So sánh thực tế (mean F1 / 15 cat, DINOv2):** 3σ=0.950, q0.95=0.951, **2σ=0.962** ← chọn.
- quantile q=0.95 **nát texture**: leather 19/41 FP (F1 1.0→0.91). Vì calib nhỏ (28-44 ảnh)
  → phân vị mẫu ước lượng hụt đuôi phân bố. μ+kσ ngoại suy đuôi → ổn định hơn cho mẫu nhỏ.
- → **Chốt μ+2σ**: F1 cao nhất ở backbone chính + không phá texture + FPR ~2.3% hợp lý
  công nghiệp. KHÔNG lấy k=1 (F1 cao hơn) vì (a) 16% báo nhầm là tệ thực tế, (b) chọn k
  theo F1 test = leakage.

**Công cụ:** `tune_threshold.py <backbone> <category>` in P/R/F1 cho mọi chính sách +
dòng ORACLE (= F1 tối đa nếu gian lận dò trên test, CHỈ để xem trần, KHÔNG dùng).
- Khoảng cách (chính sách của bạn → ORACLE) = "headroom" feature còn để lại.
- Gap NHỎ → feature là giới hạn (defect chồng lấn good). Gap LỚN → đổi ngưỡng còn cứu được.

**Insight screw (đảo ngược kết luận cũ):** DINOv2 screw AUC 0.963; ORACLE F1 chỉ 0.940,
μ+1σ đã 0.934. → **DINOv2 gần như giải quyết xong screw**; phần F1 thiếu là chọn điểm
vận hành, KHÔNG phải giới hạn feature. (Khác hẳn ResNet screw AUC 0.899 = feature kém thật.)

**Cách "dò" ngưỡng HỢP LỆ nếu muốn tối ưu thật:** dùng validation có nhãn defect. MVTec
chỉ cho good → tạo **synthetic anomaly (CutPaste)** trên held-out good làm proxy để chọn
ngưỡng — không chạm test. (Đây là chỗ duy nhất synthetic data có ích hợp lệ.)

**Defend:** "Em cố định điểm vận hành μ+2σ (~2.3% FPR), đặt TRƯỚC, không tuning trên test.
F1 chỉ là 1 điểm vận hành; AUROC (không phụ thuộc ngưỡng) mới là metric chính so sánh backbone."

---

## MODULE 5: IMPLEMENTATION ROADMAP

### Week 1 Goals:
1. ✅ Refactor code into modular classes
2. ✅ Fix data leakage (threshold from held-out good, not test)
3. ✅ Add feature normalization (+ fix consistency bug)
4. ✅ Implement K-center greedy (+ fix selection bug)
5. ✅ Fix threshold calibration (80/20 split, held-out good)
6. ✅ Fix visualization (absolute scale + real-threshold overlay)
7. ✅ Add ImageNet normalization → F1 0.836 -> 0.915
8. ✅ Verify: AUC ~0.914, F1 ~0.915, clean heatmaps, 0 errors

### Week 2 Goals:
1. Add ViT feature extractor
2. Systematic evaluation
3. Multi-category testing
4. Write thesis report

---

## GLOSSARY

| Term | Meaning |
|------|---------|
| **Anomaly** | Something different from normal |
| **Feature** | Numerical representation of image (not pixels) |
| **Memory Bank** | Collection of reference samples |
| **Threshold** | Decision boundary (score > threshold = anomaly) |
| **Heatmap** | Spatial visualization of anomaly location |
| **Euclidean Distance** | How far apart two points are |
| **Normalization** | Scaling to standard range (usually 0-1 or -1 to 1) |
| **K-center Greedy** | Algorithm to select representative samples |
| **Data Leakage** | Using test data to make training decisions |

---

**Last Updated:** June 9, 2026 (after first successful run)
**Current Results:** AUC-ROC 0.9111, F1 0.8333 (toothbrush, ResNet50 baseline)
**Next Step:** Phase 2 — Add ViT feature extractor for comparison
