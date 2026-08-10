# s2 — Opened Topic Ideas (รอบ recurrent visual attention + latent)

> **มาจาก**: [[12-recurrent-visual-attention]], [[13-latent-space-and-shortcuts]] **ต่อจาก**: [[s1-opened-topic-ideas]] **เกี่ยวข้อง**: [[11-perciever-and-more]], [[10-attention-collapse-and-field-equilibrium]], [[6-fixed-point-and-what-transformer-chasing]], [[4-hopfield-internal]]

---

## 0. การ reframe — อันนี้สำคัญกว่าตัวไอเดียทุกอัน

$$\text{เดิม: "optimize ยังไงให้โมเดลมองภาพนิ่งๆ แล้วเข้าใจ"}$$ $$\downarrow$$ $$\text{ใหม่: "optimize ยังไง ถ้า attention size เล็ก แต่โมเดลเลือกได้ว่าจะมองอะไร และจดอะไร"}$$

**นี่ไม่ใช่การเปลี่ยนวิธีแก้ปัญหา แต่เป็นการเปลี่ยนตัวปัญหา** — เป็นท่าที่ถูกต้องเวลาเจอ field ที่แออัด

ชื่อที่ควรใช้เวลาไปคุยกับคน (ให้ฟังเป็นงานวิจัย ไม่ใช่ไอเดียลอย):

$$\textbf{Bounded-bandwidth visual reasoning}$$

ประโยคขาย:

> _"ปัจจุบันเราวัดว่าโมเดลเข้าใจภาพแค่ไหนเมื่อ**ให้ดูทั้งภาพ** ผมสนใจว่ามันเข้าใจได้แค่ไหนเมื่อ**ต้องเลือกดู**"_

**ผลข้างเคียงที่ดี**: ทันทีที่ตั้งกรอบแบบนี้ งานทุกอันที่คนอื่นทำมากลายเป็น special case ($\text{attention size} = N$) แทนที่จะเป็นคู่แข่ง

---

## 1. ไอเดียหลัก ⭐⭐⭐ — latent คำนวณทั้ง $l$ (ตำแหน่ง) และ $k$ (ความละเอียด)

### 1.1 ทำไมอันนี้คือของดีที่สุด — ตารางที่บอกทุกอย่าง

|                        | ตำแหน่ง  | **ความละเอียด / scale**                               |
| ---------------------- | -------- | ----------------------------------------------------- |
| RAM 2014               | เรียนรู้ | **คงที่** (3 scale ตายตัว)                            |
| STN 2015               | เรียนรู้ | เรียนได้ในทางทฤษฎี **แต่ไม่มีใครศึกษาจริงจัง**        |
| DRAW 2015              | เรียนรู้ | **เรียน $\sigma$ ได้** ← ตัวเดียวในประวัติศาสตร์ที่ทำ |
| Deformable DETR 2021   | เรียนรู้ | **คงที่** (multi-scale ตายตัว)                        |
| ViT / ทุกอย่างปัจจุบัน | —        | **คงที่** (patch size ตายตัว)                         |

$$\boxed{\ \text{"ตรงนี้ต้องซูมแค่ไหน" เป็นคำถามที่แทบไม่มีใครให้โมเดลตอบเอง — และ DRAW ทำไว้ตั้งแต่ปี 2015 แล้วก็ไม่มีใครตามต่อ}\ }$$

### 1.2 สูตร

$$\text{latent } z_t \ \longrightarrow\ \big(l_t,\ s_t\big) \ \longrightarrow\ \rho(x, l_t, s_t) \ \longrightarrow\ z_{t+1}$$

| ตัวแปร                   | ความหมาย                      | ช่วงค่า                      |
| ------------------------ | ----------------------------- | ---------------------------- |
| $z_t \in \mathbb{R}^{d}$ | latent state ณ รอบที่ $t$     | —                            |
| $l_t \in [-1,1]^2$       | จุดศูนย์กลางที่จะมอง          | normalized coords            |
| $s_t \in (0, 1]$         | **สัดส่วนของภาพที่จะครอบ**    | 0.02 = ซูมสุด, 1.0 = ทั้งภาพ |
| $g$                      | ขนาด grid ที่อ่าน (**คงที่**) | 16×16                        |
| $\rho$                   | patch ที่อ่านได้              | $\mathbb{R}^{g^2 C}$         |

การทำนาย (แนะนำให้ทำนาย $\log s$ เพื่อให้ scale เป็น multiplicative):

$$l_t = \tanh(W_l z_t), \qquad \log s_t = W_s z_t \ \Rightarrow\ s_t = \mathrm{clip}\big(e^{W_s z_t},\ s_{\min},\ 1\big)$$

sampling grid (นี่คือ STN affine ที่ล็อคเป็น isotropic scale + translation):

$$p_{ij} = l_t + s_t \cdot (u_i,\ v_j), \qquad (u_i, v_j) \in [-1,1]^2 \text{ เป็น grid คงที่ } g\times g$$

$$\rho[i,j] = \text{bilinear}\big(x,\ p_{ij}\big)$$

**differentiable ทั้ง $l_t$ และ $s_t$:**

$$\frac{\partial \rho}{\partial l_t} = \frac{\partial \rho}{\partial p}\cdot 1, \qquad \frac{\partial \rho}{\partial s_t} = \sum_{ij}\frac{\partial \rho}{\partial p_{ij}}\cdot (u_i, v_j)$$

> gradient ของ scale คือ **ผลรวมถ่วงน้ำหนักของ image gradient โดยถ่วงด้วยระยะจากศูนย์กลาง** — ตีความได้ตรงๆ ว่า "ขอบนอกให้ข้อมูลเพิ่มไหม ถ้าไม่ ก็ซูมเข้า"

### 1.3 ⚠️ รายละเอียด implementation ที่สำคัญที่สุด — aliasing

**ถ้าไม่ทำอันนี้ ไอเดียจะพังเงียบๆ**

ถ้า $s_t$ ใหญ่ (ครอบพื้นที่กว้าง) แต่ grid ยังเป็น $g\times g$ เท่าเดิม → **sampling rate ต่ำกว่า Nyquist → aliasing** → อ่านได้ค่าขยะ และ gradient ก็ขยะตาม

**ทางแก้: อ่านจาก image pyramid (mipmap) แล้ว interpolate ข้าม level ด้วย**

$$\text{level ที่ควรใช้: } \ \ell^* = \log_2\left(\frac{s_t \cdot W}{g}\right)$$

$$\rho = (1-\alpha)\cdot\text{bilinear}\big(x^{(\lfloor \ell^_\rfloor)}, p\big) + \alpha\cdot\text{bilinear}\big(x^{(\lceil \ell^_\rceil)}, p\big), \qquad \alpha = \ell^* - \lfloor\ell^*\rfloor$$

**นี่คือ trilinear sampling** — เหมือนที่ deformable attention ทำข้าม feature map หลาย scale และเหมือนที่กราฟิกทำ mipmap มา 30 ปีแล้ว

**ผลพลอยได้ที่สำคัญมาก**: pyramid ทำให้ trade-off เป็นของจริงเชิงกายภาพ

$$s_t \text{ ใหญ่} = \text{เห็นกว้าง แต่หยาบจริงๆ} \qquad s_t \text{ เล็ก} = \text{เห็นคม แต่แคบ}$$

$$\Longrightarrow\ \textbf{โมเดลโกงด้วยการตั้ง } s_t = 1 \textbf{ ตลอดไม่ได้ เพราะมันจะเห็นแต่ภาพเบลอ}$$

โดยไม่มี pyramid โมเดลจะเรียนรู้ที่จะซูมออกให้สุดแล้วจบ — **นี่คือ shortcut ของไอเดียนี้โดยเฉพาะ** (ดู [[13-latent-space-and-shortcuts]] §6 กฎ "อะไรที่เลี่ยงได้ จะถูกเลี่ยง")

### 1.4 ทำไมมันน่าสนใจกว่าแค่ "เพิ่ม parameter อีกตัว"

$s_t$ **ไม่ใช่แค่พารามิเตอร์ มันคือการตัดสินใจเรื่องงบประมาณ**

$$\text{ซูมออก} = \textbf{สำรวจ} \text{ (explore)} \qquad\qquad \text{ซูมเข้า} = \textbf{ยืนยัน} \text{ (exploit)}$$

โมเดลต้องเรียนรู้ว่า _"ตอนนี้กูกำลังหา หรือกำลังตรวจสอบ"_ — **นี่คือพฤติกรรมที่ตีความได้และวัดได้**

$$\text{ถ้าพล็อต } s_t \text{ เทียบกับ } t \text{ แล้วเห็นเส้นลาดลง (coarse} \to \text{fine)}$$ $$\Longrightarrow \textbf{นั่นคือ coarse-to-fine ที่เกิดขึ้นเอง ไม่ได้ถูกบังคับ}$$

**รูปนี้รูปเดียวขายเปเปอร์ได้** เพราะมันคือหลักฐานว่าโมเดลเรียนกลยุทธ์ ไม่ใช่แค่ fit

และมันเชื่อมกับทฤษฎีที่มีอยู่แล้ว: coarse-to-fine = **graduated non-convexity** = วิธีข้ามกำแพงใน [[12-recurrent-visual-attention]] §12.4 → **โมเดลค้นพบวิธีแก้กำแพงด้วยตัวเอง**

### 1.5 Loss ที่ควรใช้

$$\mathcal{L} = \underbrace{\mathcal{L}_{task}}_{\text{หลัก}} + \lambda_1 \underbrace{\mathcal{L}_{loc}}_{\text{aux (optional)}} + \lambda_2 \underbrace{\mathcal{L}_{cover}}_{\text{กัน collapse}} + \lambda_3 \underbrace{\mathcal{L}_{budget}}_{\text{กัน zoom-out}}$$

| ก้อน                   | สูตร                                                            | ทำไม                                                          |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| $\mathcal{L}_{loc}$    | $\sum_t \lVert l_t - c^{gt}_t\rVert^2$ เมื่อมี bbox             | ground truth ฟรีจาก dataset — **ไม่ต้องมี teacher model**     |
| $\mathcal{L}_{cover}$  | $-\sum_{t\neq t'} \lVert l_t - l_{t'}\rVert$ หรือ entropy bonus | กัน location collapse ([[12-recurrent-visual-attention]] §17) |
| $\mathcal{L}_{budget}$ | $\sum_t s_t^2$                                                  | กันไม่ให้ซูมออกสุดตลอด                                        |

**ข้อควรระวัง**: $\mathcal{L}_{loc}$ เป็นดาบสองคม — ถ้าใส่แรงไป โมเดลจะกลายเป็น bbox regressor แล้วเลิกคิด **ควรใส่เป็น warm-up แล้ว decay ลง หรือทำ ablation ทั้งมีและไม่มี**

### 1.6 Metric ที่ต้องวัด

| Metric                                                       | บอกอะไร                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| $s_t$ vs $t$                                                 | coarse-to-fine เกิดไหม                               |
| $\mathrm{Var}_t[l_t]$                                        | location collapse ไหม                                |
| IoU($l_T$, gt box)                                           | grounding ถูกไหม                                     |
| **accuracy vs read-fraction** $\dfrac{\sum_t (s_t W)^2}{HW}$ | **"ทำได้ 90% โดยอ่านแค่ 2%"** ← กราฟที่ไม่มีใครพล็อต |
| accuracy vs $k$ (chain length)                               | E7 ใน [[13-latent-space-and-shortcuts]] §10          |

### 1.7 ความเสี่ยงของไอเดียนี้

| ความเสี่ยง                           | ทางแก้                                             |
| ------------------------------------ | -------------------------------------------------- |
| $s_t \to 1$ ตลอด (ซูมออกสุด)         | **pyramid** ทำให้เบลอจริง + $\mathcal{L}_{budget}$ |
| $s_t \to s_{\min}$ ตลอด (ซูมเข้าสุด) | จะหาไม่เจอ → task loss ลงโทษเอง                    |
| gradient ของ $s$ ระเบิด              | ทำนาย $\log s$ ไม่ใช่ $s$ + clip                   |
| coarse-to-fine ไม่เกิด               | อาจเพราะ $T$ น้อยไป หรือ task ไม่ต้องการ → ดู §3   |

---

## 2. ไอเดียรอง ⭐⭐ — retina แบบ associative retrieval (เชื่อม Hopfield)

### 2.1 ปัญหาที่แก้

_"จะรู้ได้ยังไงว่าควรมองตรงไหนต่อ"_ — โดยไม่ต้องเดินฝ่ากำแพง และไม่ต้องใช้ RL

### 2.2 สูตร

$$\text{glance ความละเอียดต่ำ} \ \longrightarrow\ M = {m_p}_{p=1}^{N_{low}} \quad \text{(associative memory)}$$ $$\text{latent } z_t \ \longrightarrow\ q_t = W_q z_t \quad \text{(query)}$$ $$l_t = \sum_{p} \mathrm{softmax}\big(\beta, \langle q_t,\ m_p\rangle\big)_p \cdot \mathrm{pos}(p) \quad \text{(retrieve \textbf{พิกัด})}$$

**นี่คือ modern Hopfield retrieval เป๊ะๆ** ([[4-hopfield-internal]], [[6-fixed-point-and-what-transformer-chasing]]) **แต่ retrieve "ที่อยู่" แทน "เนื้อหา"**

$$\text{Hopfield ปกติ: } \xi^{new} = X,\mathrm{softmax}(\beta X^\top \xi) \quad \text{— ดึงเนื้อหากลับมา}$$ $$\text{อันนี้: } l = P,\mathrm{softmax}(\beta M^\top q) \quad \text{— ดึง \textbf{พิกัด} กลับมา แล้วค่อยไปอ่านของจริงที่ความละเอียดเต็ม}$$

### 2.3 ทำไมมันสวย

| ปัญหา                 | แก้ได้ยังไง                                                   |
| --------------------- | ------------------------------------------------------------- |
| กำแพง / local minimum | **ไม่มีการเดิน — retrieve ตรงๆ**                              |
| ต้องใช้ RL ไหม        | **ไม่ — softmax differentiable**                              |
| ราคา                  | $O(N_{low})$ บนภาพย่อ **ถูกมาก** (ย่อ 8 เท่า = ถูกลง 64 เท่า) |
| ตีความได้ไหม          | **ได้ — attention map บนภาพย่อ = "มันกำลังนึกถึงอะไร"**       |

$$\boxed{\ \text{"คิดในหัว"} = \text{query เข้า associative memory ที่สร้างจาก glance}\ }$$

**นี่คือเรื่องเล่าที่รวมทุกอย่างที่เคยทำมา**: Hopfield (retrieval) + Perceiver (latent) + RAM (glimpse) — **ไม่ใช่การเริ่มใหม่ แต่คือการรวมร่าง**

### 2.4 ข้อควรระวัง

- softmax แบบนี้เป็น **soft** → ถ้า attention กระจาย $l_t$ จะไปตกตรงกลางระหว่างจุด (centroid problem) → ควรใช้ $\beta$ สูง หรือ top-1 + straight-through
- **ไม่ควรเอา retrieve ตำแหน่งเดียว** ควร retrieve $K$ ตำแหน่ง (แบบ deformable) แล้วอ่านพร้อมกัน → รักษา parallelism

---

## 3. ไอเดียที่ควร**ลดระดับ** ⚠️ — latent chain แยกส่วนประกอบภาพ (stack เล็ก → ใหญ่)

**อันนี้อ่อนที่สุดในสามอัน ขอเตือนแรงๆ**

$$\text{"ประกอบส่วนย่อยเป็นส่วนใหญ่"} = \textbf{part-whole hierarchy problem}$$

Hinton สู้กับมันมา 10 ปี (Capsule Networks → GLOM) **ยังไม่มีใครทำสำเร็จ** ปัญหาคือ:

| ปัญหา                      | รายละเอียด                                                                     |
| -------------------------- | ------------------------------------------------------------------------------ |
| ไม่มี supervision          | "ส่วนประกอบที่ถูกต้อง" ของภาพคืออะไร ไม่มีใครนิยามได้                          |
| ไม่มี metric               | วัดยังไงว่าแยกได้ดี                                                            |
| **จะกลายเป็น placeholder** | ตามกฎใน [[13-latent-space-and-shortcuts]] §6 เป๊ะๆ — ไม่มีอะไรบังคับ ก็ไม่เกิด |

### ข้อเสนอ: อย่าทำเป็นเป้าหมาย ให้เป็น**ผลพลอยได้ที่วัดทีหลัง**

$$\text{ถ้าไอเดีย 1 ทำงาน แล้วพล็อตลำดับ } (l_t, s_t) \text{ ออกมา}$$ $$\Longrightarrow \text{"ซูมออกดูโครงรวม แล้วซูมเข้าดูรายละเอียด" คือ decomposition ที่ \textbf{เกิดขึ้นเอง}}$$

**decomposition ที่เกิดเองแล้ววัดได้ มีค่ากว่า decomposition ที่บังคับแล้ววัดไม่ได้เยอะมาก**

> เก็บไอเดียนี้ไว้เป็น "analysis section" ของเปเปอร์ ไม่ใช่ "method section"

---

## 4. ไอเดีย ⭐⭐ — Pointer-chase benchmark (contribution ที่ derisk ที่สุด)

### 4.1 ทำไมต้องมี

จาก [[13-latent-space-and-shortcuts]] §8: **vision ไม่มี GSM8K** — ไม่มี benchmark ที่ยาก**แบบลึก** มีแต่ยาก**แบบกว้าง**

$$\text{ถ้าไม่มี benchmark ที่บังคับให้คิดเป็นลำดับ ก็พิสูจน์อะไรไม่ได้เลย ไม่ว่าโมเดลจะดีแค่ไหน}$$

### 4.2 การออกแบบ

```
canvas 2000×2000
ป้ายขนาด 40×40 กระจายอยู่ N ป้าย (มี decoy เยอะๆ)
ป้าย A เขียนว่า "→ B"
ป้าย B เขียนว่า "→ C"
...
ป้ายสุดท้ายมีคำตอบ
คำถามบอกแค่ป้ายเริ่มต้น
```

**นี่คือ linked list ที่ render เป็นภาพ**

### 4.3 ทำไมมันเป็นเครื่องมือที่ดี

| คุณสมบัติ                      | เหตุผล                                       |
| ------------------------------ | -------------------------------------------- |
| **ปรับ $k$ ได้**               | ได้กราฟ accuracy vs chain length ← **E7**    |
| ขนานไม่ได้เชิงทฤษฎี            | รู้ป้ายถัดไปไม่ได้จนกว่าจะอ่านป้ายปัจจุบัน   |
| decoy ฆ่า saliency shortcut    | มองที่เด่นสุดไม่ช่วยเลย                      |
| synthetic → ground truth ครบ   | รู้ว่าควรมองที่ไหน ตอนไหน → ทำ E1–E6 ได้หมด  |
| ควบคุมตัวแปรอื่นได้            | ไม่ปนกับ world knowledge หรือ language prior |
| **ตรวจ checklist 4 ข้อได้ครบ** | ดู [[13-latent-space-and-shortcuts]] §9.4    |

### 4.4 ทำไมมันคือประกันความเสี่ยง

$$\text{โมเดลพัง} \Rightarrow \text{ยังมี benchmark ที่คนอื่นใช้ได้}$$ $$\text{โมเดลเวิร์ค} \Rightarrow \text{มี benchmark ที่ทำให้ผลมีความหมาย}$$

**ชนะทั้งสองทาง** และใช้เวลา ~2 สัปดาห์ ไม่ใช่ 6 เดือน

---

## 5. ไอเดียเล็กที่เก็บไว้ก่อน

| #   | ไอเดีย                                                                    | สถานะ                                                    |
| --- | ------------------------------------------------------------------------- | -------------------------------------------------------- |
| 5.1 | **Adaptive halting** — เรียนรู้ว่าเมื่อไหร่ควรหยุดมอง                     | ทำเป็น ablation ของไอเดีย 1 ได้เลย ไม่ต้องแยกโปรเจค      |
| 5.2 | **Pointer read บน modality อื่น** (video, 3D, long document)              | deformable/IPS ติด 2D grid หมด — ว่างจริง แต่ scope ใหญ่ |
| 5.3 | **Bounded-bandwidth เป็น general backbone** (Perceiver IO + pointer read) | ว่างจริง แต่ต้องมีผลจากเฟส 1 ก่อน                        |
| 5.4 | **Self-supervised where-to-look**                                         | LTRP/LookWhere เริ่มแล้ว — ตามได้ แต่ไม่ใช่ที่ว่าง       |

### เชื่อมกับ [[s1-opened-topic-ideas]]

- **Object-bound dynamic latent** — ถ้า latent แต่ละตัวถือ 1 object แล้วแต่ละตัวมี $(l, s)$ ของตัวเอง → **นั่นคือไอเดีย 1 แบบ multi-latent** = deformable attention ที่ latent เลือก scale เอง **สองไอเดียนี้รวมร่างได้**
- **Query-conditioned compaction** — ไอเดีย 2 (associative retrieval) คือ compaction ที่ query เป็นคนเลือกพิกัด ไม่ใช่เลือก token → **เป็น special case ที่แข็งกว่า** เพราะ compaction เกิดก่อนอ่าน ไม่ใช่หลังอ่าน

---

## 6. สรุปสถานะ Open / Pending / Closed

### 🔴 Closed — อย่าเสียเวลา

| หัวข้อ                                              | ปิดโดย                                                                         |
| --------------------------------------------------- | ------------------------------------------------------------------------------ |
| hard glimpse + REINFORCE เพื่อ classification       | STN, deformable                                                                |
| patch selection บนภาพ gigapixel                     | IPS (16 gigapixel / 5GB), Attention Sampling                                   |
| sparse/deformable attention เป็นกลไกประหยัด compute | Deformable DETR — เป็นของมาตรฐานแล้ว                                           |
| zoom-in visual search บนภาพความละเอียดสูง           | V\*/SEAL, DyFo (training-free), ZoomEye, CVSearch                              |
| latent CoT ในข้อความ เวิร์คไหม                      | Coconut, CODI, Huginn + survey                                                 |
| แปะ latent visual token เข้า VLM แล้วเก็บคะแนน VQA  | **แน่นมาก** — Mirage, LVR, ILVR, Monet, LanteRn, DeepLatent, OneLatent, UniVLR |
| latent visual token ถือ visual semantics ไหม        | **ตอบแล้วว่าไม่** — ICML 2026                                                  |

### 🟡 Pending — มีคนทำอยู่ ผลยังไม่นิ่ง

| หัวข้อ                                              | สถานะ                                             |
| --------------------------------------------------- | ------------------------------------------------- |
| **ทำยังไงให้ latent visual token load-bearing**     | **เปิดกว้าง ← รอยแตกหลัก**                        |
| RL บน latent ผ่าน Gumbel/superposition              | active — Latent-SFT, Latent-GRPO, SofT-GRPO       |
| active perception เป็น Bayesian experimental design | emerging — FOVEA, perceptual bandwidth bottleneck |
| interpretability ของ latent reasoning               | early, ยังไม่มี protocol ที่ยอมรับกัน             |
| self-supervised where-to-look                       | LTRP, LookWhere                                   |

### 🟢 Open — ที่ว่างจริง (เรียงตามความคุ้ม)

| #     | ช่อง                                                       | ทำไมว่าง                                       |
| ----- | ---------------------------------------------------------- | ---------------------------------------------- |
| **1** | **latent เลือก scale/resolution เอง**                      | DRAW ทำไว้ปี 2015 **แล้วไม่มีใครตามต่อ 10 ปี** |
| **2** | **protocol ประเมิน latent visual reasoning ที่ตรวจสอบได้** | ICML เพิ่งเปิดปัญหา ยังไม่มีใครเสนอทางแก้      |
| **3** | **benchmark ที่ยากแบบลึกสำหรับ vision**                    | vision ไม่มี GSM8K                             |
| **4** | **latent สั่งพิกัดเอง differentiable ทั้งวง**              | สาย visual search ใช้ text/tool call/MCTS หมด  |
| **5** | ground truth ของ "visual CoT" คืออะไร                      | ไม่มีใครนิยาม — gaze trace? eye-tracking?      |
| 6     | pointer read นอก 2D grid                                   | scope ใหญ่                                     |

---

## 7. แผนการทำงาน (โปรเจคนอก → optimize เพื่ออัตราการเรียนรู้ ไม่ใช่ความแน่นอนของผล)

$$\text{Thesis: optimize } \textbf{ความแน่นอนของผลลัพธ์} \qquad \text{Side project: optimize } \textbf{ความรู้ต่อเวลา}$$

|                                      | thesis  | โปรเจคนอก                      |
| ------------------------------------ | ------- | ------------------------------ |
| ต้องชนะ SOTA                         | ต้อง    | **ไม่ต้อง**                    |
| ต้องมี deliverable ที่ไม่พึ่งผลลัพธ์ | ดี      | **จำเป็น**                     |
| scope                                | ใหญ่ได้ | **เล็กที่สุดที่ยังมีความหมาย** |
| ทำ dataset เอง                       | เสี่ยง  | **ได้เปรียบ — ไม่มีใครแข่ง**   |

### เฟส 0 (1–2 สัปดาห์) — หาหลักฐานว่าปัญหามีจริง

- สร้าง synthetic pointer-chase (§4): canvas 2000×2000, ป้าย 40px, $k \in {1..5}$
- ยิงเข้า Qwen-VL / LLaVA ที่มีอยู่ — **ไม่ต้องเทรนอะไรเลย**
- วัด: accuracy vs $k$ + **blind baseline** (ลบภาพทิ้ง)

> **Kill criterion**: ถ้า SOTA ทำ $k{=}4$ ได้สบาย → ปัญหาไม่มีจริง → หยุด

**เฟสนี้อย่างเดียวก็คุยกับอาจารย์ได้แล้ว** ถ้ามันตกที่ $k=2$ จริง = มีกราฟที่ทรงพลังในมือ

### เฟส 1 (1 เดือน) — โมเดลเล็กที่สุด

- glance 64×64 → Hopfield-style retrieval (§2) → $(l_t, s_t)$ (§1) → trilinear read จาก pyramid → วน
- latent = vector เดียว, $T$ glimpse, **ไม่มี LLM ไม่มี pretrain**
- baseline: ViT เต็มภาพ / glimpse สุ่ม / **parallel-$k$** ← E5 สำคัญสุด

> **Kill criterion**: ถ้า sequential $\not>$ parallel → ไม่มี reasoning เกิด → **กลับไปแก้ dataset ไม่ใช่แก้โมเดล**

### เฟส 2 (ถ้าเฟส 1 รอด)

- แปะเข้า VLM จริง / adaptive halting / ทดสอบบน V\*Bench, HR-Bench
- **อย่าเริ่มที่นี่เด็ดขาด** — debug ไม่ได้ว่าพังเพราะอะไร

---

## 8. สคริปต์คุยกับอาจารย์

**เปิดด้วยข้อสังเกต ไม่ใช่ไอเดีย:**

> _"ผมสังเกตว่า latent reasoning เวิร์คใน text แต่ ICML 2026 พิสูจน์ด้วย causal mediation ว่าไม่เวิร์คในภาพ — latent token ทำตัวเป็น placeholder_
>
> _ผมคิดว่าสาเหตุไม่ได้อยู่ที่โมเดล แต่อยู่ที่**โจทย์** — vision benchmark สืบทอดจากยุค CNN ที่ยากแบบกว้าง ไม่ใช่ยากแบบลึก โมเดลจึงไม่มีเหตุผลจะคิดเป็นลำดับ_
>
> _ผมอยากทดสอบสมมติฐานนี้ด้วยงานที่บังคับให้ลึก แล้วดูว่ากลไกแบบ recurrent visual attention กลับมามีความหมายไหม"_

**แล้วค่อยเสนอไอเดีย** — พอมีข้อสังเกตนำหน้า ไอเดียจะฟังเหมือนผลของการคิด ไม่ใช่ของที่หยิบมามั่ว

### คำถามที่ควรถามอาจารย์ (สำคัญกว่าสิ่งที่จะเล่า)

1. มี dataset ที่บังคับ referential chain อยู่แล้วไหม หรือต้องสร้างเอง
2. สร้าง synthetic benchmark เองแล้วตีพิมพ์ได้ระดับไหน (workshop / dataset track / main)
3. มี compute ให้เล่นแค่ไหน — จะได้ scope ให้พอดี
4. มีใครในแล็บทำสาย interpretability ไหม (protocol §10 ใน [[13-latent-space-and-shortcuts]] เข้ากันได้)

### สิ่งที่ต้องยอมรับก่อนโดนถาม

| ประเด็น                                     | ท่าที                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| CapImagine (text) ชนะ latent แล้ว           | **พิกัดคือ symbolic interface ที่เล็กที่สุด** — verifiable + grounded ด้วยราคา 1% ของ text |
| efficiency อย่างเดียวขายไม่ได้              | ขาย **interpretability คู่กัน**                                                            |
| สายนี้แออัดมาก                              | ต่างตรง **latent สั่งพิกัดเอง ไม่ใช่ tool call ผ่านภาษา**                                  |
| ถ้า benchmark ไม่ deep-hard จะพิสูจน์ไม่ได้ | **นี่คือเหตุผลที่ต้องทำเฟส 0 ก่อน**                                                        |

---

## 9. อันตรายที่ต้องระวังที่สุดตอนนี้

$$\boxed{\ \textbf{ทำสามไอเดียพร้อมกัน}\ }$$

เลือก **ไอเดีย 1 (latent เลือก $l$ และ $s$) + ไอเดีย 2 (associative retrieval หาพิกัด)** ทำให้จบก่อน ไอเดีย 3 → ลดเป็น analysis ไอเดีย 4 (benchmark) → **ทำก่อนทุกอย่าง** เพราะมันคือประกัน

---

## 10. TODO ที่ทำได้ทันที

- [ ] เขียน generator ของ pointer-chase dataset (canvas, ป้าย, decoy, ปรับ $k$ ได้)
- [ ] ยิงเข้า Qwen-VL / LLaVA วัด accuracy vs $k$ + blind baseline
- [ ] hook attention weight ของตำแหน่งคำตอบใน LLaVA ดูว่าตกที่ช่วง visual token เท่าไหร่ → **นี่คือ $\beta$ ที่วัดได้ด้วยโค้ด 20 บรรทัด**
- [ ] เปิด V\*Bench / GQA เลือกคำถาม 5 ข้อ มาร์คเองว่าอันไหนบังคับ closed-loop → ถ้าเกิน 3/5 ไม่บังคับ ต้องสร้าง subset เอง
- [ ] อ่าน arXiv 2602.22766 (CapImagine) ให้จบ — เป็นเปเปอร์ที่สำคัญที่สุดของสายนี้
- [ ] อ่าน arXiv 1905.03711 (Attention Sampling) — สั้น สวย ตอบคำถามเยอะ
- [ ] prototype trilinear pyramid sampler แยกเป็น module เล็กๆ ทดสอบว่า gradient ของ $s$ ไหลถูกไหม
