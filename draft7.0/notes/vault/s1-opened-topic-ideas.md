# ideas.md — dynamic latent space (image side)

> โน้ตงานของกูเอง เขียนลบได้ตามใจ คู่กับ [[11-perceiver-and-more]] (ทฤษฎีเต็ม ภาษาอังกฤษ) อัปเดตล่าสุด: หลังจบเซสชัน Perceiver / Perceiver IO
>
> **ธีมหลักที่กูไล่ตาม:** context window ที่มีขนาดจำกัด แต่รับมือข้อมูลใหญ่/ยาวได้ — เอาแนวคิด context compaction ฝั่ง LLM มาลงฝั่ง image

---

## สารบัญ

- [ภาพรวม: ปิด / ปิดครึ่ง / เปิด](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#%E0%B8%A0%E0%B8%B2%E0%B8%9E%E0%B8%A3%E0%B8%A7%E0%B8%A1-%E0%B8%9B%E0%B8%B4%E0%B8%94--%E0%B8%9B%E0%B8%B4%E0%B8%94%E0%B8%84%E0%B8%A3%E0%B8%B6%E0%B9%88%E0%B8%87--%E0%B9%80%E0%B8%9B%E0%B8%B4%E0%B8%94)
- [ข้อสรุปเชิงกลยุทธ์](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#%E0%B8%82%E0%B9%89%E0%B8%AD%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B%E0%B9%80%E0%B8%8A%E0%B8%B4%E0%B8%87%E0%B8%81%E0%B8%A5%E0%B8%A2%E0%B8%B8%E0%B8%97%E0%B8%98%E0%B9%8C)
- [IDEA 1 — Object-bound dynamic latent](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#idea-1--object-bound-dynamic-latent)
- [IDEA 2 — Query-conditioned compaction](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#idea-2--query-conditioned-compaction)
- [IDEA 3 — Two-tier read (poor read loop)](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#idea-3--two-tier-read-poor-read-loop)
- [ไอเดียสำรอง](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#%E0%B9%84%E0%B8%AD%E0%B9%80%E0%B8%94%E0%B8%B5%E0%B8%A2%E0%B8%AA%E0%B8%B3%E0%B8%A3%E0%B8%AD%E0%B8%87)
- [Experiment backlog](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#experiment-backlog)
- [Reading queue](https://claude.ai/chat/18b8e8c6-25dc-4b4a-8462-9c038a531ca3#reading-queue)

---

## ภาพรวม: ปิด / ปิดครึ่ง / เปิด

_(สถานะ ณ กลางปี 2026 — เปเปอร์ที่ลงปี 2026 ได้จากการเสิร์ช ต้องไปเช็คต้นฉบับเอง)_

### 🔒 ปิดแล้ว — อย่าเสียเวลา

| หัวข้อ                                     | สถานะ                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| cross-attn bottleneck ลด O(N²) ได้จริงมั้ย | ปิดตั้งแต่ Set Transformer (2019)                                          |
| decode ออกมาเป็นทรงอะไรก็ได้               | ปิด — output query จบเรื่อง                                                |
| weight sharing ในโครงนี้                   | ปิด — ช่วย overfit ชัด + ต้องแยกรอบแรก                                     |
| latent เก็บภาพได้แค่ไหน                    | ปิด — TiTok 32 tokens ก็ reconstruct ได้, FlexTok FID < 2 ที่ 8–128 tokens |
| Perceiver ชนะ specialist มั้ย              | ปิด — ไม่ชนะ กลายเป็น **module** (Resampler / Q-Former) ไม่ใช่ backbone    |
| ต้องมี CNN/ViT front-end มั้ย              | ปิด — ระบบจริงมีหมด                                                        |
| 1D latent ดีกว่า 2D grid มั้ย              | ปิด(ๆ) — 1D กำจัด grid redundancy ได้                                      |

> **"Perceiver ในฐานะสถาปัตยกรรมหลัก" ตายแล้ว** **"learned latent bottleneck ในฐานะชิ้นส่วน" ชนะเบ็ดเสร็จ**

### 🟡 ปิดครึ่งเดียว — มีคำตอบแต่ห่วย

**(a) Dynamic M**

มีแล้ว:

- **Dynamic Latent Perceivers** — เทรนครั้งเดียว เปลี่ยนจำนวน latent ตอน inference ได้ (speech)
- **FlexTok** — 256×256 → 1 ถึง 256 token, hierarchical + semantic, ใช้ **nested dropout** ทำให้ reconstruct ได้ทุกความยาว
- **ElasticTok** — conditioned บนเฟรมก่อนหน้า, mask สุ่มตัดท้าย, ได้ 3.5× บนภาพ / 5× บนวิดีโอ

ยังไม่ปิดตรงไหน — **AdaTok ระบุ gap ไว้ตรงๆ**: วิธีพวกนี้แสดงว่า variable-length ทำได้ แต่ยังมีช่องว่างระหว่าง **"reconstruct ได้หลายความยาว"** กับ **"ตัดสินงบเองได้ (autonomous budgeting)"**

```
ที่มี   :  โมเดล reconstruct ได้ทุกความยาว  →  แต่ "มนุษย์" เลือกความยาว
ที่ขาด  :  โมเดลดูภาพแล้วตัดสินใจเอง         →  "ภาพนี้ 12 token, ภาพนั้น 180"
```

ทำไมยาก: `M` เป็นจำนวนเต็ม (ไม่ differentiable), batch ยากเมื่อแต่ละภาพความยาวไม่เท่ากัน, ไม่มีสัญญาณกำกับว่า "พอ" คือเท่าไหร่

**(b) Latent เป็น state ข้ามเวลา (streaming)**

งานเยอะมาก แต่เป็น engineering ไม่ใช่ principle โมเดล streaming ต้องสร้าง bounded working context จาก stream ที่ไม่มีขอบเขต — วิธีที่ใช้กันคือ memory bank / event memory, KV-cache + retrieval, token pruning หรือ compression, และ recurrent/latent state

สายหลักตอนนี้คือ **training-free KV cache compression**: ใช้ cosine similarity ตัด redundancy แล้วใช้ **generic proxy query** (มักเป็น generation template) คำนวณ attention score เป็นตัวแทนความสำคัญ

> 💡 **"generic proxy query" = ยอมแพ้** — มันไม่รู้ว่าจะถูกถามอะไรเลยเดามั่วๆ **ทั้งที่ Perceiver มี learned query อยู่แล้วโดยธรรมชาติ** ← ช่องว่างชัดมาก ดู IDEA 2

**(c) Fixed point / convergence**

DEQ มีตั้งแต่ 2019 แต่แทบไม่มีใครรวมกับ Perceiver — ยัง unroll 8 รอบเหมือนเดิม เรื่อง Jacobian / spectral radius / gradient สะสม / `C₁` แยก **ยังไม่มีใครวิเคราะห์บน Perceiver โดยเฉพาะ** (ดู [[11-perceiver-and-more]] §8)

### 🟢 เปิดอยู่

1. **ใครตัดสิน `M` และตัดสินยังไง** → autonomous budgeting
2. **กฎ write / evict / consolidate / allocate / protect ของ latent** ← กว้างที่สุด
3. **latent แต่ละตัวควรหมายถึงอะไร** → binding problem
4. **multi-scale ใน latent** (FlexTok ได้ hierarchy บนแกน "ลำดับ" ไม่ใช่แกน "สเกลเชิงพื้นที่")
5. **causal / autoregressive latent สำหรับภาพ** (Perceiver AR แก้ให้ภาษาแล้ว ภาพยังไม่มี)
6. **latent ที่ generative ไม่ใช่แค่ discriminative** (IO ทำได้แค่ 20.7 PSNR) — prerequisite ของ world model

---

## ข้อสรุปเชิงกลยุทธ์

### ลำดับการแก้ที่ถูกต้อง

```
2. binding          →  latent ตัวไหนถืออะไร
3. budgeting        →  ควรมีกี่ตัว
4. eviction         →  เมื่อไหร่ลบ/รวม
```

**ต้องแก้ตามลำดับ 2 → 1 → 3** เพราะ (3) ต้องการ (2) และ (2) ทำให้ (1) มีความหมาย ถ้ากระโดดไป (1) หรือ (3) ก่อน จะติดกำแพง **"ไม่รู้จะลบอะไร"** ← นี่คือกำแพงที่งาน streaming ทั้งวงการชนอยู่ตอนนี้

### เรื่อง "ฝั่ง image ไม่มีอะไรให้จับ"

ถูกครึ่งเดียว

```
ภาษา  :  หน่วยแยกกันชัดแล้ว (คำ ประโยค ย่อหน้า turn)  →  compaction มีรอยตัดให้
ภาพ   :  ไม่มีขอบเขตในตัว pixel ต่อเนื่อง             →  ไม่มีรอยตัด   ← ถูก
```

แต่ภาพมีของที่ภาษาไม่มี:

| ภาพ/วิดีโอมี                                          | ภาษาไม่มี                       |
| ----------------------------------------------------- | ------------------------------- |
| **spatial redundancy** มหาศาล — pixel ข้างเคียงเดาได้ | ทุก token มีข้อมูล              |
| **temporal redundancy** — 30 fps = ภาพเดิม 30 ครั้ง   | ไม่มี                           |
| **physical continuity** — วัตถุไม่หายแล้วโผล่ใหม่     | ไม่มี                           |
| **object boundary** — เป็น unit จริง แค่ต้องหาเอง     | คำเป็น unit แต่ไม่ใช่ "สิ่งของ" |

**และประเด็นชี้ขาด:**

```
ภาพนิ่ง  :  ไม่มีขอบธรรมชาติ  →  unit ต้องเป็น "วัตถุ"        (แกนพื้นที่)
วิดีโอ   :  มีขอบธรรมชาติ    →  unit เป็น shot / scene / event  (แกนเวลา)
```

Temporal Perceiver แบ่ง latent query เป็น 2 ชนิด — **boundary** กับ **context** — เพื่อจัดการ temporal redundancy ของวิดีโอ = goal/fact split เวอร์ชันที่มีคนทำแล้ว

งานล่าสุดก็ไปทางเดียวกัน: ObjectStream ใช้ **object เป็น memory anchor** (object history ถาวร + การเปลี่ยนแปลงระดับ object + บริบทภาพล่าสุด) ภายใต้งบ token จำกัด

> **สรุป: ปัญหาจริงของกูคือ unit discovery ไม่ใช่ compaction** พอมี unit แล้ว compaction เหมือนกันทั้งสองฝั่ง

### ตารางต้นทุนที่ต้องจำ

| operator                   | ต้นทุน/element | ลด N ได้   | เลือกตามเนื้อหา? | ใช้กับอะไร    |
| -------------------------- | -------------- | ---------- | ---------------- | ------------- |
| strided conv / patchify    | `k²·C` ≈ 2e3   | มาก (256×) | ❌               | grid          |
| pooling                    | ~0             | มาก        | ❌               | grid          |
| local / window attention   | `w·d` ≈ 5e4    | กลาง       | ✅ ในหน้าต่าง    | grid, seq     |
| cross-attn (learned query) | `M·d` ≈ 5e5    | ทุกอัตรา   | ✅ ทั่วโลก       | **อะไรก็ได้** |
| full self-attention        | `N·d` ≈ 5e7    | ❌         | ✅               | อะไรก็ได้     |

**กฎ: ไล่จากบนลงล่าง ใช้ตัวถูกที่สุดที่ยังทำงานได้**

---

# IDEA 1 — Object-bound dynamic latent

> **ความคุ้ม: ⭐⭐⭐ สูงสุด** — แก้ 3 open problem พร้อมกัน (binding + budgeting + hierarchy) และ binding คือ prerequisite ของทุกอย่างที่เหลือ

## 1.1 ปัญหาที่จะแก้

Perceiver ไม่มีอะไรบังคับให้ latent แยกหน้าที่กัน:

```
Z ∈ ℝ^(M×D)  →  ไม่มี constraint ใดๆ ว่า Z[3] กับ Z[47] ต้องต่างกัน
                → อาจซ้ำกัน อาจเกาะของเด่นๆ เหมือนกันหมด อาจกระจายมั่ว
```

**ผลที่ตามมา:**

- ลบ latent ตัวไหนก็ไม่รู้ว่าจะเสียอะไร → **evict ไม่ได้**
- ไม่รู้ว่าตัวไหนซ้ำกับตัวไหน → **consolidate ไม่ได้**
- ไม่รู้ว่ากี่ตัวถึงพอ → **budget ไม่ได้**

> **ทุกอย่างที่กูอยากทำ ติดที่ข้อเดียวกัน: latent ไม่มีความหมายที่ระบุได้**

## 1.2 กุญแจทางเทคนิค: เปลี่ยนแกน softmax

นี่คือรายละเอียดที่กูคิดว่าเป็นแกนกลางของไอเดียนี้ทั้งหมด

```
Perceiver:
    A = softmax_over_N( Q Kᵀ )          A ∈ ℝ^(M×N), แต่ละแถวรวม = 1
    → latent แต่ละตัวแจกงบ 100% ของตัวเองบน pixel
    → latent ไม่แข่งกัน
    → ทุกตัวไปเกาะของเด่นๆ เหมือนกันได้ ไม่มีใครห้าม

Slot Attention:
    A = softmax_over_M( Q Kᵀ )          A ∈ ℝ^(M×N), แต่ละ**คอลัมน์**รวม = 1
    → pixel แต่ละตัวแจกงบ 100% ของตัวเองบน slot
    → slot ต้องแข่งกันแย่ง pixel
    → เกิด object binding เอง โดยไม่ต้อง supervise
```

**เปลี่ยนแกนเดียว = ได้ competition = ได้ object-centric representation**

Slot Attention เต็มรูป (Locatello et al. 2020) มี 3 ชิ้น:

```python
# 1. attention แข่งกัน (softmax over slots)
attn = softmax(k @ q.T / sqrt(d), dim=-1)          # (N, M) normalize บนแกน M

# 2. weighted mean normalization (กัน slot ที่ชนะเยอะเกินจะ dominate)
attn = attn / attn.sum(dim=0, keepdim=True)        # normalize บนแกน N อีกที
updates = attn.T @ v                                # (M, d)

# 3. GRU update แทน residual add
slots = GRU(updates, slots)
slots = slots + MLP(LN(slots))
```

**ข้อ 2 สำคัญมาก** — ถ้าไม่มี slot ที่กินพื้นที่เยอะจะได้ update ใหญ่กว่าเพื่อนตามขนาด พอ normalize แล้ว slot ที่ดูแล 5 pixel กับ slot ที่ดูแล 5000 pixel ได้ update ขนาดเทียบเท่ากัน

**ข้อ 3** GRU ให้ gating — slot เลือกได้ว่าจะรับ update แค่ไหน (ต่างจาก `Z ← Z + ...` ที่รับหมด) → นี่คือ **protect** ตัวแรกที่กูต้องการ

## 1.3 สถาปัตยกรรมที่เสนอ

```
X (image)
  │
  ├─ CNN/ViT front-end  →  X̃ ∈ ℝ^(N'×C')        [ลด N แบบไม่เลือก, ราคาเกือบฟรี]
  │
  ├─ Binding stage      →  S ∈ ℝ^(M_max × D)     [Slot Attention, softmax over M]
  │     วน 3 รอบตามสูตร Slot Attention           [แข่งกัน → แต่ละ slot จับของหนึ่งชิ้น]
  │
  ├─ Halting            →  ตัดสินว่าใช้จริงกี่ slot  [ACT-style, differentiable]
  │
  ├─ Processing         →  self-attn ลึกๆ บน slot ที่รอด
  │
  └─ Decode             →  output query cross-attend เข้า slot
```

### ทำไมต้องมี CNN front-end

- Slot Attention ต้นฉบับรันบน CNN feature map อยู่แล้ว ไม่ได้รันบน raw pixel
- ลด `N` ทำให้ binding stage ถูกลง 256 เท่า → วนหลายรอบได้
- object boundary มองเห็นชัดกว่าใน feature space มากกว่า pixel space

### ทำไมต้องเป็น `M_max` แล้วค่อยตัด ไม่ใช่ `M` ผันแปรตรงๆ

- batch ได้ (ทุก sample มี `M_max` slot เท่ากัน แค่ mask ต่างกัน)
- differentiable (soft mask ไม่ใช่ hard cut)
- ตอน inference ค่อย hard-cut ตาม threshold เพื่อประหยัดจริง

## 1.4 กลไก halting (ส่วนที่เป็นของใหม่)

ยืม ACT (Graves 2016) มาใช้กับ **จำนวน slot** แทน **จำนวนชั้น**

```
สำหรับ slot m:
    h_m = σ( w_h · LN(S[m]) + b_h )     ∈ (0,1)      halting score
    เรียง slot ตาม h จากมากไปน้อย (หรือใช้ลำดับ fix แบบ nested dropout)

    cumulative:  c_m = Σ_{j≤m} h_j
    mask:        α_m = clamp(1 − relu(c_m − 1), 0, 1)      ← soft, differentiable

    S_used[m] = α_m · S[m]
```

- `α_m = 1` สำหรับ slot แรกๆ, ค่อยๆ ลดเป็น 0 หลังจาก cumulative ถึง 1
- **จำนวน slot ที่ใช้จริง** ≈ `Σ_m α_m` — เป็นจำนวนจริง differentiable

**Budget loss:**

```
L_budget = λ_b · Σ_m α_m           ← กดดันให้ใช้ slot น้อย
L_total  = L_task + L_recon + L_budget
```

`λ_b` คุม trade-off ระหว่าง "ใช้ slot น้อย" กับ "ทำงานได้ดี" → **ปรับ λ_b แล้ววาด Pareto frontier: จำนวน slot vs คุณภาพ** ← นี่คือรูปหลักของเปเปอร์

**ทางเลือกอื่นที่ควรลอง:**

| วิธี                               | ข้อดี                              | ข้อเสีย                         |
| ---------------------------------- | ---------------------------------- | ------------------------------- |
| ACT halting (ข้างบน)               | soft, ง่าย, มี precedent           | อาจ collapse ไปใช้ slot เดียว   |
| nested dropout (FlexTok)           | ได้ ordering ฟรี, พิสูจน์แล้วบนภาพ | ยังต้องมีคนเลือกความยาว         |
| Gumbel top-k                       | เลือกได้ตรงๆ                       | variance สูง                    |
| ทำนายจำนวนก่อนเลย (predictor head) | ถูกที่สุด                          | ต้องมี label ว่า "ควรใช้กี่ตัว" |

> กูว่าท่าที่ดีที่สุดคือ **ACT + nested dropout ผสมกัน**: nested dropout ให้ ordering (slot แรกสำคัญสุด) แล้ว ACT ตัดสินว่าตัดตรงไหน

## 1.5 Loss ทั้งหมด

```
L = L_task                              งานปลายทาง
  + λ_r · L_recon                       บังคับให้ slot เก็บข้อมูลจริง (สำคัญมาก ดู 1.7)
  + λ_b · Σ_m α_m                       budget penalty
  + λ_e · L_entropy                     กัน slot collapse (ดูล่าง)
```

**`L_entropy` — กัน degenerate solution:**

ปัญหาที่จะเจอแน่นอน: slot ทุกตัวจับ "ทั้งภาพ" เหมือนกันหมด แล้วก็ยัง reconstruct ได้ วิธีกัน: บังคับให้ attention map ของแต่ละ slot **คม** และ **ไม่ทับกัน**

```
คม     :  minimize  H(A[:, m])  = −Σₙ A[n,m] log A[n,m]     ต่อ slot
ไม่ทับ  :  minimize  Σ_{m≠m'} ⟨A[:,m], A[:,m']⟩              cross-slot overlap
```

## 1.6 Metric ที่ต้องวัด

```
1. slot count distribution     →  ภาพซับซ้อนใช้ slot เยอะจริงมั้ย?
                                  วาด scatter: (จำนวน object จริง) vs (Σα)
                                  ถ้าไม่ correlate = binding ไม่เกิด

2. ARI / mIoU ของ attention map เทียบกับ ground-truth segmentation
                               →  slot จับ object จริงมั้ย (ใช้ CLEVR / MOVi ก่อน)

3. slot swap test              →  สลับ slot m ของภาพ A กับภาพ B
                                  ถ้า output เปลี่ยนแบบ "วัตถุนั้นถูกแทนที่" = binding จริง
                                  ถ้าเละไปหมด = ไม่ได้ binding

4. slot drop test              →  ลบ slot m ทิ้ง แล้วดูว่าเสียอะไร
                                  ถ้าเสีย "วัตถุหนึ่งชิ้น" = ดี
                                  ถ้าเสียคุณภาพทั่วภาพ = ไม่ได้ binding    ← ตัวชี้ขาด

5. Pareto: (Σα) vs task metric →  รูปหลักของเปเปอร์
```

> **ตัวที่ 4 คือตัวชี้ขาดที่สุด** เพราะมันวัด "evict ได้มั้ย" โดยตรง ถ้า drop slot แล้วเสียแค่วัตถุชิ้นเดียว = กูมี building block ของ compaction แล้ว

## 1.7 ความเสี่ยง / จุดที่จะพัง

| ความเสี่ยง                          | อาการ                                          | ทางแก้                                                                                  |
| ----------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| **slot collapse**                   | ทุก slot เหมือนกันหมด                          | entropy loss + cross-slot penalty + init ที่ต่างกัน                                     |
| **binding พังบนภาพจริง**            | Slot Attention เวิร์คบน CLEVR แต่ ImageNet ยาก | เริ่มจาก CLEVR/MOVi → ค่อยขึ้นภาพจริง; ใช้ DINO feature ที่มี object structure อยู่แล้ว |
| **halting collapse**                | `Σα → 1` เสมอ (ใช้ slot เดียว)                 | warmup `λ_b` จาก 0, หรือใส่ minimum slot                                                |
| **reconstruction ไม่พอเป็น signal** | slot เก็บ texture ไม่เก็บ object               | ใช้ perceptual loss / DINO feature loss แทน pixel MSE                                   |
| **batching ช้า**                    | mask ไม่ช่วยประหยัดจริงตอน train               | ยอมรับ — ประหยัดจริงตอน inference เท่านั้น                                              |

> **จุดที่กูคิดว่าเสี่ยงที่สุดคือแถวที่ 2** — Slot Attention ขึ้นชื่อว่าเวิร์คสวยบน synthetic แต่พอเจอภาพธรรมชาติแล้วพัง งานหลังๆ แก้ด้วยการใช้ **DINO/DINOv2 feature เป็น input** เพราะ self-supervised feature พวกนี้มี object structure ฝังอยู่แล้ว → **ถ้าจะทำ ให้เริ่มจาก DINO feature ไม่ใช่ CNN ที่เทรนเอง**

## 1.8 Milestone

```
M0  reproduce Slot Attention บน CLEVR                        [1 สัปดาห์] baseline
M1  เปลี่ยน input เป็น DINOv2 feature, ทดสอบบน COCO           [2 สัปดาห์] binding บนภาพจริง
M2  ใส่ halting + budget loss, วัด Pareto                     [2 สัปดาห์] dynamic M
M3  slot drop / swap test                                     [1 สัปดาห์] พิสูจน์ว่า evict ได้
M4  ต่อ Perceiver IO decoder เข้าไป, ทดสอบ dense task         [2 สัปดาห์] ครบวง
M5  ขยายไปวิดีโอ (slot ต่อเนื่องข้ามเฟรม)                      [?]        → IDEA 2
```

**หยุดได้ที่ M3** — ถ้า drop test ผ่าน แปลว่าไอเดียเวิร์ค ที่เหลือเป็น engineering

## 1.9 งานที่ต้องอ่าน/เทียบ

```
Slot Attention (Locatello 2020)          ต้นฉบับ — อ่านก่อนสุด
SAVi / SAVi++                            slot บนวิดีโอ
DINOSAUR                                 slot + self-supervised feature ← สำคัญ
Object-Centric Learning survey           ภาพรวมว่าอะไรพังบนภาพจริง
FlexTok                                  nested dropout + ordering
AdaTok (2606.07185)                       autonomous budgeting — งานใกล้ที่สุด
ACT (Graves 2016)                        halting mechanism
```

---

# IDEA 2 — Query-conditioned compaction

> **ความคุ้ม: ⭐⭐ ช่องว่างชัดที่สุด เขียนเปเปอร์ได้เร็วที่สุด** ไม่ต้องรอ binding ก็เริ่มได้ (แต่ถ้ามี binding จะดีกว่ามาก)

## 2.1 ปัญหาที่จะแก้

งาน streaming/long-video ทั้งวงการต้องตัดสินใจว่า **"เก็บอะไร ทิ้งอะไร"** ภายใต้งบจำกัด

**วิธีที่ใช้กันอยู่ตอนนี้ทั้งหมดมี 2 แบบ และห่วยทั้งคู่:**

```
แบบ A — redundancy-based
    keep_score(x) = 1 − max cosine_sim(x, สิ่งที่เก็บไว้แล้ว)
    "เก็บของที่ไม่ซ้ำ"
    ปัญหา: ท้องฟ้าที่ไม่ซ้ำกันเลย ก็ได้คะแนนสูง ทั้งที่ไม่มีใครสนใจ

แบบ B — proxy-query attention
    ใช้ generation template เป็น query ปลอม แล้ววัด attention score
    "เก็บของที่ template สนใจ"
    ปัญหา: template ไม่ใช่คำถามจริง มันคือการเดา
```

> **ทั้งสองแบบมีปัญหาเดียวกัน: มันไม่รู้ว่าจะถูกถามอะไร เลยเดา**

## 2.2 ข้อสังเกตที่เป็นแกนของไอเดีย

**Perceiver มี learned query อยู่แล้ว และมันคือคำตอบของคำถามข้างบนพอดี**

```
Z₀ ∈ ℝ^(M×D)  =  M คำถามที่โมเดลเรียนรู้มาจาก data ว่า "อะไรควรถามกับโลกใบนี้"
```

`Z₀` **ไม่ใช่ template ที่มนุษย์เขียน** และ **ไม่ใช่ heuristic** — มันคือ distribution ของคำถามที่ถูก learn มาจาก task จริง

**เพราะฉะนั้น:**

```
keep_score(x) = max_m  ⟨ Z₀[m] W_Q , x W_K ⟩ / √d

แปลว่า: "สิ่งนี้ตอบคำถามที่โมเดลสนใจได้มั้ย"
ไม่ใช่:  "สิ่งนี้ซ้ำกับเพื่อนมั้ย"
```

> **นี่คือความต่างเชิงหลักการ: เปลี่ยนจาก compression criterion (redundancy) เป็น relevance criterion (query-conditioned utility)**

## 2.3 สถาปัตยกรรมที่เสนอ

```
stream:  X₁ → X₂ → X₃ → ...            (เฟรม / chunk / segment)

memory:  Z ∈ ℝ^(M×D)                   ขนาดคงที่ ไม่โต

ต่อ chunk t:
    1. ENCODE   X̃_t = front-end(X_t)
    2. SCORE    s_n = max_m ⟨Z[m]W_Q, X̃_t[n]W_K⟩ / √d      ต่อทุก element ใน chunk
    3. ADMIT    เลือก top-k จาก X̃_t ตาม s                   ← query-conditioned
    4. WRITE    Z ← update(Z, X̃_t[selected])
    5. EVICT    ถ้า Z เต็ม → ตัดสินใจว่าจะทับ slot ไหน
```

**ข้อ 3 กับ 5 คือของใหม่ ที่เหลือมีคนทำแล้ว**

### กฎ eviction ที่เสนอ

Perceiver มีแค่ `write` แบบบวกทับ ต้องเพิ่ม 4 อย่าง:

```
utility(m)   =  EMA ของ attention mass ที่ slot m ได้รับ
                จากทั้ง (a) การ query ของ decoder และ (b) การ score ใน step 2
                → slot ที่ไม่มีใครใช้ ค่าจะตกลงเรื่อยๆ

age(m)       =  จำนวน chunk ตั้งแต่ slot m ถูกเขียนล่าสุด

evict score  =  utility(m) − γ · age(m)          ← ต่ำสุดโดนทับ

consolidate  =  ถ้า cos(Z[i], Z[j]) > τ  →  merge เป็นตัวเดียว
                Z[i] ← (n_i·Z[i] + n_j·Z[j]) / (n_i + n_j)     น้ำหนักตามจำนวนครั้งที่ถูกเขียน
                → ปล่อย slot j ว่าง

protect      =  slot ที่ utility สูงกว่า threshold → gate ปิด รับ update ไม่ได้
                (ทำด้วย GRU gate หรือ scalar gate ต่อ slot)
```

## 2.4 ปัญหาใหญ่: ทำให้มัน differentiable ยังไง

top-k กับ evict เป็น discrete หมด มี 3 ทาง:

**(a) Soft everything (ง่ายสุด, แนะนำเริ่มที่นี่)**

```
แทนที่จะ hard select → ใช้ soft weight
    w_n = σ( (s_n − threshold) / temperature )
    write ด้วยน้ำหนัก w_n

แทนที่จะ hard evict → decay
    Z[m] ← (1 − ε_m) · Z[m] + ε_m · new_content
    โดย ε_m = softmax over m ของ (−evict_score)
```

ข้อดี: gradient ไหลได้หมด | ข้อเสีย: ไม่ได้ประหยัดจริงตอน train

**(b) Straight-through**

hard ตอน forward, soft ตอน backward — ประหยัดจริงแต่ gradient bias

**(c) Auxiliary supervision (ท่าที่กูชอบที่สุด)**

```
เทรน scorer แยกด้วย distillation จาก dense oracle:

1. รัน dense model (ไม่ compact) บน chunk → ได้ attention mass จริงต่อ element
2. เทรน scorer ให้ทำนาย attention mass นั้น
   L_score = ‖ s_n − attention_mass_n^dense ‖²
3. ตอน deploy ใช้ scorer อย่างเดียว

→ scorer เป็น "ตัวย่อ" ของ dense attention
→ ได้ทั้ง differentiable และประหยัด
```

## 2.5 การทดลองที่พิสูจน์ไอเดีย

**การเปรียบเทียบหลัก — ทุกตัวใช้งบ memory เท่ากันเป๊ะ:**

| baseline            | keep criterion                   |
| ------------------- | -------------------------------- |
| random              | สุ่ม                             |
| uniform / stride    | ทุก k เฟรม                       |
| redundancy (cosine) | ไม่ซ้ำกับที่เก็บไว้              |
| proxy-query         | template attention               |
| **ของกู**           | **learned `Z₀` query attention** |

**สิ่งที่ต้องวัด:**

```
1. accuracy vs memory budget curve         ← กราฟหลัก
2. needle-in-haystack: ยัดของสำคัญไว้ที่เฟรมสุ่ม แล้วถามทีหลัง
   → วัด recall ตามตำแหน่งของ needle
   → นี่คือการทดสอบว่า "เก็บถูกตัวมั้ย" โดยตรง
3. ทดสอบ query ที่ไม่เคยเห็นตอนเทรน
   → ถ้ายังดี = Z₀ generalize จริง ไม่ได้ overfit คำถาม
4. ablation: ใช้ Z₀ ตอน init vs Z ที่อัปเดตแล้ว เป็นตัว score
   → ตัวหลังคือ "compaction ที่รู้บริบท" น่าจะดีกว่า แต่ต้องพิสูจน์
```

**ข้อ 3 คือจุดตายที่ต้องระวัง** — ถ้า `Z₀` เก่งเฉพาะคำถามที่เคยเทรน ไอเดียก็ไม่มีค่า ต้องออกแบบ eval ให้มี query distribution shift ตั้งแต่แรก

## 2.6 ทำไมกูคิดว่าอันนี้เขียนเปเปอร์ได้เร็ว

- **ไม่ต้อง train from scratch** — เอา Perceiver Resampler / Q-Former ที่ pretrain แล้วมาใช้
- **baseline ชัด มีคนทำไว้เยอะแล้ว** — เทียบง่าย
- **contribution ระบุได้ในประโยคเดียว**: _"แทนที่ proxy query ด้วย learned query"_
- **eval มีมาตรฐานอยู่แล้ว** — long video QA benchmark เพียบ

## 2.7 ความเสี่ยง

| ความเสี่ยง                                 | ทางแก้                                                               |
| ------------------------------------------ | -------------------------------------------------------------------- |
| `Z₀` เอนเอียงตาม task ที่ pretrain มา      | ทดสอบ query shift ตั้งแต่ต้น; อาจต้อง train `Z₀` ด้วย query หลากหลาย |
| ไม่ชนะ redundancy baseline                 | ผสมกัน: `score = α·relevance + (1−α)·(1−redundancy)` แล้ว ablate `α` |
| eviction ทำให้เกิด catastrophic forgetting | ใส่ protect gate + วัด recall ของเฟรมเก่าโดยเฉพาะ                    |
| `M` คงที่ไม่พอสำหรับวิดีโอยาวมาก           | ผสมกับ IDEA 1 → `M` ผันแปรตามความซับซ้อนของ scene                    |

---

# IDEA 3 — Two-tier read (poor read loop)

> **ความคุ้ม: ⭐ ทำง่ายสุด ใช้พิสูจน์ concept ได้เร็ว** — รายละเอียดเต็มอยู่ใน [[11-perceiver-and-more]] §14

สรุปสั้น:

```
X   = raw input 50,176 × 261         แพง ละเอียด
X̃   = CNN/ViT feature 196 × 768      ถูก หยาบ

รอบ 1     :  full read จาก X       →  สร้าง Z แบบ degree of freedom สูง
รอบ 2-8   :  poor read จาก X̃       →  conditional query สำหรับ reasoning
decode    :  output query พก X มา  →  รายละเอียดส่งตรงถึงปลายทาง
```

**ราคา: แพงกว่า Perceiver IO แค่ ~3%, ถูกกว่า Perceiver เดิม 7.8 เท่า**

**หลักการที่อยู่เบื้องหลัง:**

> **"ละเอียด" กับ "ซ้ำหลายครั้ง" ไม่เคยต้องมาคู่กัน**
>
> - อ่านละเอียด → ต้องการแค่ครั้งเดียว (ตอนต้น + ตอนปลาย)
> - อ่านซ้ำ → ต้องการแค่หยาบ เพราะมันคือการ **ค้นหา** ไม่ใช่การ **ดึงรายละเอียด**

**การทดลอง 30 นาทีที่พิสูจน์ได้เลย (ไม่ต้อง train):**

```python
Z = Z0
for t in range(8):
    src = X_full if t == 0 else X_downsampled_by_k
    Z = cross[t](Z, src); Z = latent_transformer(Z)

# sweep k = 1,2,4,8,16 แล้วดู accuracy
# ถ้าแบนถึง k=8  →  ไอเดียถูก
# ถ้าตกที่ k=2   →  ต้องใช้ sparse-sharp แทน coarse-dense
```

**คู่กับการวัด attention entropy ต่อรอบ:**

```
H_t = −Σₙ A_t[m,n] log A_t[m,n]     เฉลี่ยข้าม m

H₁ สูง แล้ว H₂..H₈ ต่ำ  →  รอบหลังอ่านแค่ไม่กี่จุดอยู่แล้ว → sparse ได้ ✓
H เท่ากันหมด            →  รอบหลังยังอ่านกระจาย → sparsify แล้วเจ็บ ✗
```

**ระวัง: X̃ จะกลายเป็นลูกน้องประจบ** — gradient จาก loop ดัด CNN ให้ผลิต feature ที่ "ทำให้ loss ต่ำ" ไม่ใช่ "ซื่อสัตย์ต่อภาพ" → ใส่ reconstruction loss กัน

---

## ไอเดียสำรอง

### 4. Differentiable scale selection

ให้ latent เลือกเองว่ารอบนี้จะอ่านชั้นไหนของ feature pyramid

```
C₃ (3,136) / C₄ (784) / C₅ (196)  →  concat เป็น key/value ก้อนเดียว (N' = 4,116)
→ ให้ attention ตัดสินใจเองว่าคำถามแบบไหนควรถามที่สเกลไหน
```

ได้ multi-scale โดยไม่ต้องบังคับ — Perceiver ทำไม่ได้เพราะมีสเกลเดียว **ทำให้ coarse-to-fine เป็นสิ่งที่โมเดลเรียนรู้เอง แทนที่จะเป็นตารางที่กูเขียนตายตัว**

### 5. DEQ Perceiver

เอา Perceiver มาทำเป็น fixed-point จริงๆ:

- แก้ `Z* = f(Z*; X)` ด้วย root-finder
- backprop ผ่าน implicit function theorem
- memory O(1) ไม่ขึ้นกับจำนวนรอบ
- **ต้องแก้ปัญหา `C₁` ก่อน** (ทำให้ `Z₀` มีบริบท หรือใส่ gate) ไม่งั้นไม่มี fixed point เดียว

น่าจะเป็นเปเปอร์ analysis มากกว่า architecture — และทำได้ด้วย compute น้อย

### 6. Slot-level KV cache สำหรับ video

รวม IDEA 1 + IDEA 2:

- slot ผูกกับ object
- object หายไปจากเฟรม → slot นั้น age เพิ่ม → โดน evict
- object กลับมา → slot ใหม่ หรือ re-bind กับ slot เดิม? ← **re-identification problem**

อันนี้คือ IDEA 1 + 2 บนวิดีโอ = ปลายทางที่กูอยากไปจริงๆ แต่ต้องทำ 1 กับ 2 ให้เสร็จก่อน

### 7. Learned tokenizer-free unit discovery

ถ้าปัญหาจริงคือ unit discovery (ตามข้อสรุปข้างบน) — แล้ว unit ที่ดีที่สุดคืออะไร?

- object (spatial)
- event/shot (temporal)
- หรืออะไรที่ยังไม่มีชื่อ?

**คำถามเชิงทฤษฎี: unit ที่ดีคือ unit ที่ทำให้ description length สั้นที่สุด (MDL) หรือเปล่า** → ถ้าใช่ ก็ตั้ง objective เป็น MDL ตรงๆ แล้วให้ unit โผล่มาเอง

---

## Experiment backlog

เรียงตาม (คุ้ม / ต้นทุน)

```
[ ] E1  วัด attention entropy ต่อรอบบน Perceiver ที่ pretrain แล้ว       30 นาที  ⭐⭐⭐
[ ] E2  downsample-at-inference sweep (k = 1,2,4,8,16)                  1 ชม.    ⭐⭐⭐
[ ] E3  cache K,V แล้ววัด wall-clock ที่ประหยัดได้จริง                    1 ชม.    ⭐⭐
[ ] E4  log ‖g_t‖ และ cos(g_t, g_t') ของ shared block ระหว่าง train      2 ชม.    ⭐⭐⭐
[ ] E5  reproduce Slot Attention บน CLEVR                              1 สัปดาห์ ⭐⭐⭐
[ ] E6  slot drop test บน Slot Attention ที่ reproduce ได้              2 ชม.    ⭐⭐⭐
[ ] E7  Slot Attention บน DINOv2 feature, ทดสอบ COCO                   2 สัปดาห์ ⭐⭐
[ ] E8  Z₀-based keep score vs cosine baseline บน long-video QA         1 สัปดาห์ ⭐⭐
```

**E1, E2, E4 ทำได้ทันทีและตอบคำถามเยอะมาก — เริ่มจากตรงนี้**

---

## Reading queue

```
[ ] Slot Attention (Locatello 2020)         ← อ่านก่อนสุด สั้น เปลี่ยนมุมมองเยอะ
[ ] DINOSAUR                                 slot + self-supervised feature
[ ] Set Transformer (ISAB)                   ต้นตระกูลจริงของ latent bottleneck
[ ] FlexTok (2502.13967)
[ ] ElasticTok
[ ] AdaTok (2606.07185)                      ← ระบุ gap ที่กูจะแก้ ตรงที่สุด
[ ] Temporal Perceiver (2203.00307)          boundary/context split บนวิดีโอ
[ ] Perceiver AR
[ ] DEQ (Bai 2019)
[ ] ObjectStream (2607.28312)
[ ] Compressive Transformer
[ ] Perceiver Resampler (Flamingo paper §2)
```

---

## หมายเหตุถึงตัวเองในอนาคต

1. **binding มาก่อนเสมอ** ถ้าติดว่า "ไม่รู้จะลบอะไร" แปลว่ายังไม่มี binding
2. **อย่ากด `d` ตอนทำ poor read** มันฆ่า conditional query ซึ่งเป็นเหตุผลเดียวที่อ่านซ้ำ
3. **วัด cos(g_t, g_t') ก่อนโทษ loop** ถ้าติดลบแปลว่า share weight ผิดที่ ไม่ใช่ loop พัง
4. **shared param มี effective LR สูงกว่า 2.6–7 เท่า** เปลี่ยนโครงแล้วต้องจูน LR ใหม่
5. เปเปอร์ปี 2026 ในโน้ตนี้ได้จากการเสิร์ช **ยังไม่ได้อ่านต้นฉบับ** ต้องไปเช็คก่อนอ้าง
