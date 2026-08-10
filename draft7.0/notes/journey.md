
ตอนเเรกผมพิสูจน์ linear regression บน deep learning จาก 0 โดยไม่ใช้เเม้เเต่เมทริก เพื่อหาว่าจริงๆเเล้ว ตัวเลขมันฉลาดขึ้นได้ยังไง จากนั้นจึงไปพิสูจน์ hessian เเละทำโมเดลจริงขึ้นมาจากเเค่ NumPy เเละจึงไปศึกษาเรื่อง double descend ต่อผ่าน concept noise เเบบเเข็ง เเละเเบบอ่อน


หลักจากที่ผมพิสูจน์เเละเข้าใจ LR, CNN, RNN, GRU, LSTM, GNN เเละ transformer เบื้องต้น

ในตอนเเรกตัวผมเองคิดว่าจริงๆเเล้ว gradient descend มันผิด มันไม่ใช่สิ่งที่ neural network จริงๆเป็น ตัว axon ในระบบประสาทมันเรียนรู้เเบบ local โดยมันไม่ต้องใช้ derivative 

ผมจึงหันกลับไปศึกษาเรื่อง deep learning ที่ไม่ใช่ neural network ผมทำพยายามออกเเบบ analog circuit เอง เพื่อใช้ capacitor กับ inductor เพื่อเลียนเเบบ mechanism ของ axon จริงๆ ผ่านทฤษฎีการเรียนรู้เเบบ attribution-based learning

```
[Controller]   Brainstem  ←→  central control, loss compute, clock management
                  │
                  ↓ global broadcast bus
[Top level]    Limbic Loop  ←→  recurrent system (Cortex + Hippocampus + Commissure)
                  │
                  ↓ per-Column local bus
[Level 4]      Lobe         ←→  multi-branch DAG composition of Columns
                  │
                  ↓ Lobe-local bus
[Level 3]      Column       ←→  sequential chain of Ganglia with translate ALUs
                  │
                  ↓ Column-local bus
[Level 2]      Ganglion     ←→  hardwired 2-3-3-2 atom of computation
                  │
                  ↓ Ganglion-local bus
[Level 1]      Scap         ←→  atom of storage (one synapse weight)

```

เเต่ว่าไอเดียก็พังลง เพราะว่าสุดท้ายเเล้วถึงมันไม่ใช่ gradient เเต่ว่ามันก็ติด summation wall เหมือนกัน เเละต้องคำนวณ contribution กลับเเบบ backward ก็คือมันทำตัวเหมือน backprop ของ gradient เลย เเค่เเย่กว่าในการ train 

จากนั้นผมจึงลงไปศึกษาใหม่ให้ลึกกว่าเดิมว่า axon จริงๆเราทำงานยังไง โดยเฉพาะ concept "fire together, wire together" โดยรอบนี้ที่สนใจเป็นหลักคือ Hebbian rule โดยเฉพาะ Hinton forward forward ที่เป็นการเรียนรู้ผ่านการทำ forward ทีเดียวได้ สามารถ train model เเละทำนายคำตอบได้พร้อมกัน

ผมได้หยิบเอา 
- **Self-Contrastive Forward-Forward** algorithm มา ตัวมันใช้ $x_{pos}$ กับ $x_{neg}$ มาทำ forward ทีเดียว เเละรู้ direction การอัปเดทเลยว่าต้องปรับ w ยังไงให้ค่า goodness มันมากขึ้น
- ปรับเเต่งสมการ goodness ผ่าน InfoNCE theory ที่เเต่ละ neural ยังเรียนรู้เเบบ local เเต่ว่าสามารถมองไกลได้เป็น k layer 
- โดย concept หลักคือ ใช้ Forward Forward model ที่ไม่ต้องรอ backward เเละสามารถเรียนรู้ตลอดเวลาได้ 80% เเล้วใช้ decode model อื่นอีก 20% ซึ่งจากผลการทดลองใช้ SLDA (no gradient descend ทั้งหมด)

เเต่ว่าหลังจากทำเสร็จ ผมก็เห็นว่าจริงๆ ที่เราต้องใช้ gradient ไม่ใช่เพราะว่ามันจริง เเต่เป็นเพราะว่าเราไม่สามารถจำลอง neural network ในธรรมชาติเเบบ 1:1 ได้เลย สิ่งที่ gradient กับ SGD ทำคือวิธีที่มีประสิทธิภาพมากที่สุดในการกระจาย direction ย้อนกลับไปให้ w ทุกตัวใน model

เเละสุดท้าย ผมก็ยังพบว่าตัว 80% SCFF + 20% SLDA ของผม สุดท้ายมันก็ยังไม่ได้ solve อะไร ผมได้มาเเค่ continuous learning สำหรับ linear regression ยังไม่ได้ mechanism หลักของสมองมา

ในตอนนี้ผมเลยเข้าไปศึกษาลึกกว่าเดิมใน topics เกี่ยวกัง associate memory จาก transformer, hopfield, visual transformer, perciever, percieverIO, recurrent attention, etc. จนได้ไอเดีย **Bounded-bandwidth visual reasoning** มา

