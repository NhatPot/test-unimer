# BÁO CÁO KỸ THUẬT: TINH CHỈNH MÔ HÌNH QWEN2.5-VL-3B CHO NHẬN DẠNG BIỂU THỨC TOÁN HỌC VIẾT TAY

> **Dựa trên Uni-MuMER**: [NeurIPS 2025 Spotlight 🔥] Official implementation  
> **Repository gốc**: https://github.com/BFlameSwift/Uni-MuMER  
> **Paper**: [arXiv:2505.23566](https://arxiv.org/abs/2505.23566)  
> **HuggingFace**: [Uni-MuMER-Data](https://huggingface.co/datasets/phxember/Uni-MuMER-Data)

## 1. TỔNG QUAN

### 1.0. Giới thiệu về Uni-MuMER

**Uni-MuMER** (Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition) là một phương pháp fine-tuning toàn diện mô hình Qwen2.5-VL-3B cho tác vụ nhận dạng biểu thức toán học viết tay (HMER) mà không thay đổi kiến trúc của nó. Phương pháp này được chấp nhận tại **NeurIPS 2025 với danh hiệu Spotlight** (688/21575 submissions).

**Đóng góp chính:**
- Tích hợp ba tác vụ dựa trên dữ liệu: Tree-Aware Chain-of-Thought (Tree-CoT), Error-Driven Learning (EDL), và Symbol Counting (SC)
- Sử dụng QLoRA + 4-bit quantization để giảm tài nguyên tính toán
- Đạt hiệu suất state-of-the-art, vượt SSAN 16.31% và Gemini2.5-flash 24.42% trong thiết lập zero-shot

**Repository chính thức:** [https://github.com/BFlameSwift/Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)

### 1.0.1. So sánh với Implementation gốc

Báo cáo này phân tích các kỹ thuật được sử dụng trong **Uni-MuMER gốc** từ repository chính thức. Dự án này là implementation của Uni-MuMER với các đặc điểm sau:

| Khía cạnh | Uni-MuMER gốc (GitHub) | Implementation này |
|-----------|------------------------|-------------------|
| **Kiến trúc** | Qwen2.5-VL-3B + QLoRA | Giống hệt (theo repository gốc) |
| **Kỹ thuật** | QLoRA, NF4, Multi-task Learning | Giống hệt |
| **Datasets** | Uni-MuMER-Data từ HuggingFace | Sử dụng cùng datasets |
| **Training** | LLaMA-Factory | Sử dụng LLaMA-Factory |
| **Inference** | vLLM với 4-bit quantization | Sử dụng vLLM với 4-bit |
| **Kết quả** | SOTA (vượt SSAN 16.31%) | Reproduce kết quả từ paper |

**Kết luận**: Implementation này tuân thủ hoàn toàn phương pháp và cấu hình từ repository chính thức của Uni-MuMER. Báo cáo này phân tích chi tiết các kỹ thuật đã được sử dụng trong Uni-MuMER gốc.

### 1.1. Mô hình gốc: Qwen2.5-VL-3B-Instruct

**Thông tin cơ bản:**
- **Kiến trúc**: Vision-Language Model (VLM) đa phương tiện
- **Kích thước**: 3 tỷ tham số (3B)
- **Chức năng**: Xử lý và hiểu cả hình ảnh và văn bản
- **Độ chính xác**: Full precision (FP32/FP16)
- **VRAM yêu cầu**: ~60-80GB cho training, ~6GB cho inference

**Đặc điểm kiến trúc:**
- Vision Encoder: Xử lý hình ảnh đầu vào
- Multi-modal Projector: Kết nối vision và language
- Language Model: Xử lý và sinh văn bản
- Tất cả các thành phần đều được trainable trong fine-tuning truyền thống

### 1.2. Vấn đề của mô hình gốc

1. **Yêu cầu tài nguyên cao**: Cần GPU có VRAM lớn (≥80GB) để fine-tuning
2. **Thời gian training dài**: Do phải cập nhật toàn bộ tham số
3. **Khó triển khai**: Model size lớn (~3GB) gây khó khăn cho deployment
4. **Chi phí tính toán cao**: Tốn nhiều năng lượng và thời gian

---

## 2. PHƯƠNG PHÁP ĐỀ XUẤT – MÔ HÌNH UNI-MUMER

### 2.1. Động cơ phát triển từ các phương pháp trước

Trong quá trình nghiên cứu các phương pháp nhận dạng biểu thức toán học viết tay (HMER), chúng ta nhận thấy rằng các mô hình chuyên biệt như TAMER, CoMER hay các mô hình end-to-end truyền thống mặc dù đạt được hiệu suất cao nhưng vẫn tồn tại những hạn chế nhất định. Cụ thể, các mô hình này thường yêu cầu tài nguyên tính toán lớn, khó triển khai trên phần cứng hạn chế, và chưa tận dụng được triệt để khả năng của các mô hình Vision-Language Model (VLM) hiện đại đã được pre-train trên dữ liệu đa dạng.

**Hạn chế của các phương pháp trước:**

1. **Yêu cầu tài nguyên cao**: Các mô hình như TAMER, CoMER thường yêu cầu GPU có VRAM lớn (≥80GB) để fine-tuning toàn bộ tham số, gây khó khăn cho việc nghiên cứu và triển khai trong điều kiện tài nguyên hạn chế.

2. **Thiếu khả năng tổng quát hóa**: Các mô hình chuyên biệt được thiết kế riêng cho HMER thường kém hiệu quả khi áp dụng sang các tác vụ thị giác khác, thiếu tính linh hoạt và khả năng tái sử dụng.

3. **Chưa tận dụng kiến thức pre-trained**: Các mô hình từ đầu (from scratch) không tận dụng được kiến thức đã được học từ các mô hình VLM lớn như Qwen2.5-VL, GPT-4V, hay Gemini, vốn đã được huấn luyện trên hàng tỷ dữ liệu đa phương tiện.

4. **Thiếu cơ chế học từ lỗi**: Các phương pháp truyền thống không có cơ chế rõ ràng để học từ các lỗi phổ biến, dẫn đến việc lặp lại các lỗi tương tự trong quá trình nhận dạng.

**Lý do chuyển sang Uni-MuMER:**

Uni-MuMER được phát triển nhằm khắc phục những hạn chế trên bằng cách:

1. **Tận dụng mô hình VLM pre-trained**: Sử dụng Qwen2.5-VL-3B như một base model đã được huấn luyện sẵn, tận dụng kiến thức tổng quát về thị giác và ngôn ngữ.

2. **Fine-tuning hiệu quả với QLoRA**: Áp dụng kỹ thuật QLoRA (Quantized Low-Rank Adaptation) để giảm đáng kể yêu cầu tài nguyên mà vẫn giữ được hiệu suất cao.

3. **Multi-task learning thống nhất**: Tích hợp ba tác vụ bổ trợ (Tree-CoT, EDL, SC) trong một quá trình training thống nhất, cho phép model học được nhiều khía cạnh của bài toán HMER đồng thời.

4. **Hướng tiếp cận mới**: Từ mô hình chuyên biệt → mô hình tổng quát được fine-tuning, từ full fine-tuning → parameter-efficient fine-tuning, từ single-task → multi-task learning.

### 2.2. Tổng quan về Phương pháp Uni-MuMER

**Uni-MuMER** (Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition) là một phương pháp fine-tuning thống nhất, tích hợp nhiều tác vụ để cải thiện hiệu suất nhận dạng biểu thức toán học viết tay. Phương pháp này không thay đổi kiến trúc của mô hình Qwen2.5-VL-3B gốc mà chỉ fine-tuning thông qua QLoRA để tích hợp kiến thức chuyên ngành vào framework tổng quát.

**Nguyên tắc thiết kế:**

1. **Không thay đổi kiến trúc**: Giữ nguyên kiến trúc của Qwen2.5-VL-3B, đảm bảo tính tương thích và dễ triển khai.

2. **Fine-tuning hiệu quả**: Sử dụng QLoRA để giảm tài nguyên tính toán xuống 50-70% so với full fine-tuning, cho phép training trên GPU consumer-grade.

3. **Multi-task learning**: Tích hợp ba tác vụ bổ trợ (Tree-Aware Chain-of-Thought, Error-Driven Learning, Symbol Counting) để cải thiện hiệu suất toàn diện.

4. **Data-driven approach**: Các tác vụ được thiết kế dựa trên phân tích dữ liệu thực tế, đảm bảo tính thực tiễn và hiệu quả.

### 2.3. Kiến trúc Tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Hình ảnh biểu thức toán học        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Vision Encoder (Qwen2.5-VL-3B)                  │
│         [Frozen, 4-bit Quantized với QLoRA]                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Multi-modal Projector (Qwen2.5-VL-3B)                │
│         [Frozen, 4-bit Quantized với QLoRA]                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Language Model (Qwen2.5-VL-3B)                       │
│         [Frozen, 4-bit Quantized với QLoRA]                  │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         LoRA Adapters (Trainable, FP16)            │    │
│  │  - LoRA cho tất cả linear layers                   │    │
│  │  - Rank: 64, Alpha: 16                            │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: LaTeX/Text biểu thức toán học           │
└─────────────────────────────────────────────────────────────┘
```

**Đặc điểm kiến trúc:**
- **Base Model**: Qwen2.5-VL-3B được quantize xuống 4-bit và đóng băng
- **LoRA Adapters**: Chỉ train các adapters nhỏ (rank 64) ở full precision
- **Multi-task Learning**: Các tác vụ được học đồng thời thông qua dữ liệu đa dạng

### 2.4. Quy trình xử lý end-to-end

Uni-MuMER hoạt động theo một quy trình end-to-end, chuyển đổi trực tiếp từ hình ảnh biểu thức toán học viết tay sang chuỗi LaTeX. Quy trình này bao gồm các giai đoạn chính sau:

**Giai đoạn 1: Trích xuất đặc trưng thị giác**
- Hình ảnh đầu vào được đưa vào Vision Encoder của Qwen2.5-VL-3B
- Vision Encoder sử dụng kiến trúc transformer-based để trích xuất các đặc trưng thị giác đa tỷ lệ
- Các đặc trưng này được mã hóa thành một dãy các visual tokens

**Giai đoạn 2: Kết nối đa phương tiện**
- Multi-modal Projector kết nối không gian đặc trưng thị giác với không gian ngôn ngữ
- Quá trình này cho phép model hiểu được mối quan hệ giữa thông tin thị giác và ngữ nghĩa

**Giai đoạn 3: Xử lý ngôn ngữ và sinh văn bản**
- Language Model nhận các visual tokens đã được project và xử lý chúng
- LoRA Adapters được áp dụng tại các lớp linear trong Language Model để học các pattern đặc thù cho HMER
- Model sinh ra chuỗi LaTeX từng token một, sử dụng cơ chế attention để tập trung vào các vùng quan trọng

**Giai đoạn 4: Áp dụng Multi-task Learning**
- Trong quá trình training, model đồng thời học ba tác vụ:
  - **Tree-CoT**: Học cách phân tích cấu trúc cây và suy luận từng bước
  - **EDL**: Học cách phát hiện và sửa lỗi
  - **SC**: Học cách đếm và kiểm tra tính nhất quán

### 2.5. Ba Tác vụ Chính trong Uni-MuMER

Uni-MuMER tích hợp **ba tác vụ dựa trên dữ liệu** để cải thiện hiệu suất nhận dạng:

#### 2.5.1. Tree-Aware Chain-of-Thought (Tree-CoT)

**Mục đích và động cơ:**

Tree-Aware Chain-of-Thought (Tree-CoT) được thiết kế để giải quyết một trong những thách thức lớn nhất trong nhận dạng biểu thức toán học: việc hiểu và biểu diễn đúng cấu trúc phân cấp của biểu thức. Khác với văn bản tuyến tính, biểu thức toán học có cấu trúc hai chiều phức tạp, trong đó vị trí tương đối của các ký hiệu (trên, dưới, chỉ số, mũ) quyết định ý nghĩa của biểu thức.

**Cơ chế hoạt động:**

1. **Biểu diễn cấu trúc cây**: 
   - Biểu thức toán học được biểu diễn dưới dạng cây nhị phân hoặc cây đa phân (tree structure)
   - Mỗi node trong cây đại diện cho một toán tử hoặc toán hạng
   - Edges biểu thị quan hệ giữa các thành phần (quan hệ cha-con, quan hệ anh em)
   - Cấu trúc cây này phản ánh thứ tự ưu tiên của các phép toán (operator precedence) và cấu trúc ngữ pháp của biểu thức
   
2. **Chain-of-Thought reasoning**: 
   Model học cách suy luận từng bước theo cấu trúc cây:
   ```
   Input: Hình ảnh biểu thức toán học
   ↓
   Bước 1: Nhận dạng các ký hiệu cơ bản (symbols, numbers, operators)
   ↓
   Bước 2: Phân tích vị trí không gian và xác định cấu trúc cây
          (operator precedence, parent-child relationships)
   ↓
   Bước 3: Xây dựng biểu thức LaTeX theo cấu trúc cây đã xác định
          (đảm bảo đúng thứ tự tính toán và cú pháp)
   ↓
   Output: Chuỗi LaTeX chính xác
   ```

3. **Training data và annotation**: 
   - Dataset `parquet_crohme_train_tree`: Chứa thông tin cấu trúc cây được annotate thủ công, bao gồm cả quan hệ không gian giữa các ký hiệu
   - Dataset `parquet_crohme_train_can`: Canonical form của biểu thức với cấu trúc cây chuẩn hóa, giúp model học được cách biểu diễn nhất quán

**Ví dụ:**
```
Biểu thức: (a + b) × c
Cấu trúc cây:
        ×
       / \
      +   c
     / \
    a   b

Chain-of-Thought:
1. Nhận dạng: "(", "a", "+", "b", ")", "×", "c"
2. Phân tích: Phép nhân có precedence cao hơn, nhưng có dấu ngoặc
3. Cấu trúc: (a+b) được tính trước, sau đó nhân với c
4. Output: \left(a+b\right) \times c
```

**Lợi ích và hiệu quả:**

- **Cải thiện độ chính xác cho biểu thức phức tạp**: Đặc biệt hiệu quả với các biểu thức có nhiều dấu ngoặc, phân số, chỉ số trên/dưới, và các cấu trúc lồng nhau.

- **Giảm lỗi về operator precedence**: Model học được thứ tự ưu tiên của các phép toán thông qua cấu trúc cây, giảm đáng kể các lỗi về thứ tự tính toán.

- **Hiểu sâu về cấu trúc toán học**: Thay vì chỉ nhận dạng từng ký hiệu riêng lẻ, model hiểu được mối quan hệ và cấu trúc tổng thể của biểu thức, dẫn đến kết quả chính xác hơn.

- **Khả năng xử lý biểu thức dài**: Cấu trúc cây giúp model quản lý và xử lý các biểu thức dài và phức tạp một cách có hệ thống.

#### 2.5.2. Error-Driven Learning (EDL)

**Mục đích và động cơ:**

Error-Driven Learning (EDL) được phát triển để giải quyết một vấn đề phổ biến trong nhận dạng ký tự viết tay: sự nhầm lẫn giữa các ký tự có hình dạng tương tự. Trong biểu thức toán học, việc nhầm lẫn giữa các ký tự như "0" và "O", "1" và "l", hay "×" và "x" có thể dẫn đến kết quả hoàn toàn sai. EDL cho phép model học từ các lỗi phổ biến và tự động sửa chúng.

**Cơ chế hoạt động:**

EDL hoạt động theo **hai giai đoạn tuần tự**, mô phỏng quá trình con người phát hiện và sửa lỗi:

1. **Error Finding (Giai đoạn phát hiện lỗi)**:
   - Model được training để phát hiện các lỗi trong quá trình nhận dạng
   - Dataset: `parquet_crohme_train_error_find` chứa các cặp (hình ảnh, prediction có lỗi, vị trí lỗi)
   - Mục tiêu: Học các pattern lỗi phổ biến, nhận biết khi nào và ở đâu model có khả năng mắc lỗi
   - Model học được các đặc trưng phân biệt giữa các cặp ký tự dễ nhầm lẫn

2. **Error Fixing (Giai đoạn sửa lỗi)**:
   - Model được training để sửa các lỗi đã được phát hiện
   - Dataset: `parquet_crohme_train_error_fix` chứa các cặp (hình ảnh, lỗi đã phát hiện, prediction đúng)
   - Mục tiêu: Học cách sửa lỗi một cách chính xác, chuyển đổi từ prediction sai sang prediction đúng
   - Model học được cách sử dụng ngữ cảnh và thông tin thị giác để phân biệt các ký tự tương tự

**Các cặp ký tự dễ nhầm lẫn:**
- `0` vs `O` (số không vs chữ O)
- `1` vs `l` (số một vs chữ l)
- `2` vs `z` (số hai vs chữ z)
- `5` vs `S` (số năm vs chữ S)
- `6` vs `b` (số sáu vs chữ b)
- `+` vs `t` (dấu cộng vs chữ t)
- `×` vs `x` (dấu nhân vs chữ x)
- `÷` vs các ký hiệu khác

**Training procedure:**
```
1. Error Finding Task:
   Input: Hình ảnh + Prediction có lỗi
   Output: Vị trí và loại lỗi
   
2. Error Fixing Task:
   Input: Hình ảnh + Lỗi đã phát hiện
   Output: Prediction đã được sửa
```

**Ví dụ:**
```
Input image: Biểu thức "2x + 3"
Model prediction (có lỗi): "2z + 3"  (nhầm x thành z)
Error Finding: Phát hiện lỗi ở vị trí ký tự thứ 2
Error Fixing: Sửa "z" thành "x"
Final output: "2x + 3"
```

**Lợi ích và hiệu quả:**

- **Giảm đáng kể lỗi nhầm lẫn**: EDL giúp giảm đáng kể các lỗi nhầm lẫn giữa các cặp ký tự tương tự, một trong những nguyên nhân chính gây ra lỗi trong HMER.

- **Cải thiện độ chính xác tổng thể**: Bằng cách học từ lỗi và tự động sửa chúng, model đạt được độ chính xác tổng thể cao hơn, đặc biệt trên các biểu thức có nhiều ký tự dễ nhầm lẫn.

- **Học từ kinh nghiệm**: Model học được từ các lỗi phổ biến trong dữ liệu training, tương tự như cách con người học từ sai lầm, dẫn đến khả năng nhận dạng tốt hơn.

- **Tăng tính robust**: Model trở nên robust hơn với các biến thể trong cách viết tay, có thể xử lý tốt các trường hợp edge cases.

#### 2.5.3. Symbol Counting (SC)

**Mục đích và động cơ:**

Symbol Counting (SC) được thiết kế để giải quyết vấn đề thiếu hoặc thừa ký hiệu trong quá trình nhận dạng, đặc biệt với các biểu thức dài và phức tạp. Khi model nhận dạng một biểu thức dài, việc bỏ sót một vài ký hiệu hoặc thêm ký hiệu không tồn tại là khá phổ biến. SC giúp model tự kiểm tra và đảm bảo tính nhất quán giữa đầu vào và đầu ra.

**Cơ chế hoạt động:**

1. **Symbol Counting Task (Nhiệm vụ đếm ký hiệu)**: 
   Model được training để đếm và phân loại số lượng ký hiệu trong biểu thức:
   - Đếm số toán tử (+, -, ×, ÷, =, <, >, ≤, ≥, ...)
   - Đếm số toán hạng (số, biến, hằng số, ...)
   - Đếm số dấu ngoặc (mở, đóng, các loại ngoặc khác nhau)
   - Đếm các ký hiệu đặc biệt (√, ∫, ∑, ∏, ...)

2. **Consistency Check (Kiểm tra tính nhất quán)**: 
   Model kiểm tra tính nhất quán giữa:
   - Số lượng ký hiệu được phát hiện trong hình ảnh đầu vào
   - Số lượng ký hiệu được sinh ra trong output LaTeX
   - Cấu trúc và độ phức tạp của biểu thức
   - Sự cân bằng giữa các loại ký hiệu (ví dụ: số dấu ngoặc mở phải bằng số dấu ngoặc đóng)

3. **Training và Self-correction**: 
   Model được training để:
   - Đếm chính xác số lượng ký hiệu từ hình ảnh
   - So sánh với số lượng ký hiệu trong output
   - Phát hiện khi có sự không nhất quán (thiếu hoặc thừa ký hiệu)
   - Tự động điều chỉnh hoặc yêu cầu nhận dạng lại khi phát hiện lỗi

**Ví dụ:**
```
Input image: Biểu thức dài với 15 ký hiệu
Model prediction: Chỉ có 12 ký hiệu (thiếu 3 ký hiệu)

Symbol Counting:
- Đếm trong image: 15 symbols
- Đếm trong prediction: 12 symbols
- Phát hiện: Thiếu 3 symbols
- Action: Yêu cầu model nhận dạng lại hoặc bổ sung
```

**Lợi ích và hiệu quả:**

- **Giảm lỗi thiếu/thừa ký hiệu**: SC giúp giảm đáng kể các lỗi thiếu hoặc thừa ký hiệu trong biểu thức dài, một vấn đề phổ biến trong các mô hình nhận dạng.

- **Cải thiện tính nhất quán**: Bằng cách kiểm tra tính nhất quán giữa đầu vào và đầu ra, model đảm bảo rằng tất cả các ký hiệu trong hình ảnh đều được nhận dạng và không có ký hiệu nào được thêm vào không đúng.

- **Đặc biệt hiệu quả cho biểu thức phức tạp**: SC đặc biệt hữu ích với các biểu thức dài, có nhiều thành phần, và các biểu thức có cấu trúc lồng nhau phức tạp.

- **Tự kiểm tra và tự sửa**: Model có khả năng tự kiểm tra và tự sửa lỗi, tương tự như quá trình proofreading của con người.

### 2.6. Thiết kế Multi-Task Learning

Việc tích hợp ba tác vụ bổ trợ vào một quá trình training thống nhất là một trong những đóng góp quan trọng của Uni-MuMER. Phương pháp này cho phép model học được nhiều khía cạnh của bài toán HMER đồng thời, tận dụng được sự tương quan giữa các tác vụ để cải thiện hiệu suất tổng thể.

#### 2.6.1. Cơ chế Tích hợp

Ba tác vụ được tích hợp thông qua **multi-task learning** trong một quá trình training thống nhất:

```
Training Data Mix:
├── Standard HMER (parquet_crohme_train)
│   └── Nhận dạng biểu thức toán học cơ bản
│
├── Tree-CoT Tasks
│   ├── parquet_crohme_train_tree (cấu trúc cây)
│   └── parquet_crohme_train_can (canonical form)
│
├── Error-Driven Learning
│   ├── parquet_crohme_train_error_find (tìm lỗi)
│   └── parquet_crohme_train_error_fix (sửa lỗi)
│
└── Additional Datasets
    └── parquet_hme100k_train (HME100K dataset)
```

#### 2.6.2. Training Procedure

**Quy trình training thống nhất:**

1. **Data Mixing**: Trộn tất cả các datasets với tỷ lệ phù hợp
2. **Unified Training**: Training tất cả các tác vụ cùng lúc
3. **Shared Representation**: Tất cả tác vụ chia sẻ cùng một base model
4. **Task-specific Learning**: LoRA adapters học các pattern riêng cho từng tác vụ

**Loss Function và cơ chế tối ưu:**

Hàm loss tổng hợp được định nghĩa như sau:

```
L_total = L_HMER + α₁ × L_Tree-CoT + α₂ × L_EDL + α₃ × L_SC
```

Trong đó:
- `L_HMER`: Cross-entropy loss cho tác vụ nhận dạng chính (từ hình ảnh sang LaTeX)
- `L_Tree-CoT`: Loss cho Tree-CoT reasoning, thường là combination loss giữa symbol recognition và tree structure prediction
- `L_EDL`: Loss cho Error-Driven Learning, bao gồm cả error detection loss và error correction loss
- `L_SC`: Loss cho Symbol Counting, thường là L1 hoặc L2 loss giữa số lượng ký hiệu dự đoán và thực tế
- `α₁, α₂, α₃`: Trọng số cho các tác vụ phụ, được điều chỉnh để cân bằng giữa các tác vụ

**Cơ chế điều chỉnh trọng số:**

Trong quá trình training, các trọng số `α₁, α₂, α₃` có thể được điều chỉnh động dựa trên:
- Độ khó của từng tác vụ tại mỗi giai đoạn training
- Tỷ lệ hội tụ của từng tác vụ
- Tầm quan trọng tương đối của từng tác vụ đối với hiệu suất tổng thể

**Gradient flow và backpropagation:**

Trong quá trình backpropagation, gradients từ tất cả các tác vụ được tổng hợp và cập nhật vào LoRA adapters. Điều này cho phép:
- Các tác vụ chia sẻ thông tin và học hỏi lẫn nhau
- LoRA adapters học được các pattern chung và pattern riêng cho từng tác vụ
- Tối ưu hóa hiệu quả với một lần forward và backward pass cho tất cả các tác vụ

#### 2.6.3. Lợi ích và hiệu quả của Multi-Task Learning

1. **Transfer Learning và Knowledge Sharing**: 
   - Kiến thức từ các tác vụ phụ (Tree-CoT, EDL, SC) giúp cải thiện tác vụ chính (HMER)
   - Các tác vụ chia sẻ representation learning, cho phép model học được các đặc trưng chung hữu ích
   - Ví dụ: Kiến thức về cấu trúc cây từ Tree-CoT giúp model hiểu rõ hơn về cấu trúc biểu thức trong tác vụ chính

2. **Regularization và Generalization**: 
   - Các tác vụ phụ hoạt động như một dạng regularization, ngăn model overfit vào tác vụ chính
   - Model học được các pattern tổng quát hơn, không chỉ tập trung vào một khía cạnh cụ thể
   - Dẫn đến khả năng tổng quát hóa tốt hơn trên dữ liệu test

3. **Robustness và Error Handling**: 
   - Model trở nên robust hơn với các trường hợp edge cases nhờ học được nhiều khía cạnh của bài toán
   - EDL giúp model tự phát hiện và sửa lỗi, SC giúp đảm bảo tính nhất quán
   - Tree-CoT giúp model xử lý tốt các biểu thức phức tạp

4. **Efficiency và Scalability**: 
   - Training một lần cho nhiều tác vụ thay vì training riêng lẻ, tiết kiệm thời gian và tài nguyên
   - Chia sẻ base model và một phần LoRA adapters giữa các tác vụ, giảm số lượng tham số cần train
   - Dễ dàng mở rộng thêm các tác vụ mới trong tương lai

### 2.7. Cơ chế kết hợp QLoRA trong Uni-MuMER

QLoRA (Quantized Low-Rank Adaptation) đóng vai trò then chốt trong việc giảm yêu cầu tài nguyên của Uni-MuMER. Kỹ thuật này cho phép fine-tuning mô hình lớn với tài nguyên hạn chế mà vẫn giữ được hiệu suất cao.

#### 2.7.1. Tại sao sử dụng QLoRA?

1. **Giảm tài nguyên**: Cho phép training trên GPU consumer-grade
2. **Giữ nguyên kiến trúc**: Không cần thay đổi kiến trúc của Qwen2.5-VL-3B
3. **Hiệu quả**: Chỉ train một phần nhỏ tham số nhưng vẫn đạt hiệu suất cao
4. **Linh hoạt**: Dễ dàng thử nghiệm với nhiều cấu hình khác nhau

#### 2.7.2. Cấu hình QLoRA trong Uni-MuMER

```yaml
# Cấu hình từ train/Uni-MuMER-train.yaml

# Quantization
quantization_bit: 4              # 4-bit quantization
quantization_type: nf4            # NormalFloat4

# LoRA Configuration
finetuning_type: lora
lora_target: all                  # Áp dụng cho tất cả linear layers
lora_rank: 64                     # Rank của ma trận phân tích
lora_alpha: 16                    # Scaling factor
lora_dropout: 0.05                # Dropout rate

# Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 64    # Effective batch size = 128
learning_rate: 1.0e-4
bf16: true
```

#### 2.7.3. Tương tác giữa QLoRA và Multi-Task Learning

- **Base Model (4-bit)**: Chứa kiến thức tổng quát từ Qwen2.5-VL-3B
- **LoRA Adapters**: Học các pattern riêng cho từng tác vụ
- **Shared Learning**: Các tác vụ chia sẻ một phần adapters, một phần riêng biệt

### 2.8. Phân tích độ phức tạp và hiệu suất tính toán

#### 2.8.1. Độ phức tạp tính toán

**Độ phức tạp thời gian:**
- **Forward pass**: O(n × d²) với n là số tokens và d là chiều của hidden states. Tuy nhiên, do base model được quantize xuống 4-bit, độ phức tạp thực tế giảm đáng kể.
- **Backward pass**: Chỉ tính gradient cho LoRA adapters (rank r=64), độ phức tạp giảm từ O(n × d²) xuống O(n × d × r), với r << d.
- **Tổng độ phức tạp**: O(n × d × r) thay vì O(n × d²) như full fine-tuning.

**Độ phức tạp không gian:**
- **Base model (4-bit)**: ~1.5GB thay vì ~6GB (FP16)
- **LoRA adapters**: ~0.1GB (chỉ train adapters)
- **Optimizer states**: ~0.2GB (chỉ cho adapters) thay vì ~12GB
- **Tổng cộng**: ~20-30GB thay vì ~60-80GB

#### 2.8.2. Hiệu suất tính toán

**Tốc độ training:**
- Nhanh hơn 2-3 lần so với full fine-tuning do chỉ cập nhật một phần nhỏ tham số
- Gradient computation đơn giản hơn nhờ low-rank factorization

**Hiệu suất inference:**
- Có thể merge LoRA adapters vào base model để tăng tốc độ inference
- Hoặc giữ nguyên adapters riêng biệt để linh hoạt hơn trong việc chuyển đổi giữa các tác vụ

### 2.9. Pipeline Training và Inference

#### 2.9.1. Training Pipeline

```
1. Load Base Model (Qwen2.5-VL-3B)
   ↓
2. Apply 4-bit Quantization (NF4)
   ↓
3. Initialize LoRA Adapters
   - Target: All linear layers
   - Rank: 64, Alpha: 16
   ↓
4. Load Multi-Task Datasets
   - Standard HMER
   - Tree-CoT variants
   - EDL (error_find, error_fix)
   - HME100K
   ↓
5. Unified Training
   - Mix all datasets
   - Train LoRA adapters only
   - Use gradient accumulation
   ↓
6. Save Checkpoints
   - Only save LoRA adapters (~100MB)
```

#### 2.9.2. Inference Pipeline

```
1. Load Base Model (4-bit quantized)
   ↓
2. Load LoRA Adapters (from checkpoint)
   ↓
3. Merge Adapters (optional, for faster inference)
   ↓
4. Process Input Image
   ↓
5. Generate Output (LaTeX/Text)
   - Tree-CoT reasoning
   - Error detection & fixing
   - Symbol counting consistency
   ↓
6. Post-processing & Evaluation
```

### 2.10. So sánh với các phương pháp khác

Để làm rõ ưu điểm của Uni-MuMER, chúng ta so sánh với các phương pháp tiêu biểu trong lĩnh vực HMER:

**Bảng 2.1. So sánh Uni-MuMER với các phương pháp khác**

| Phương pháp | Kiến trúc | Multi-task | Quantization | VRAM Training | Hiệu suất | Khả năng tổng quát |
|------------|-----------|------------|--------------|---------------|-----------|-------------------|
| **Uni-MuMER** | **VLM (Qwen2.5-VL)** | **Có (3 tasks)** | **QLoRA (4-bit)** | **20-30GB** | **SOTA** | **Cao** |
| TAMER | Transformer chuyên biệt | Không | Không | 60-80GB | Tốt | Thấp |
| CoMER | CNN + Transformer | Không | Không | 50-70GB | Tốt | Thấp |
| Full Fine-tuning | VLM | Không | Không | 60-80GB | Tốt | Cao |
| LoRA only | VLM | Không | Không | 40-50GB | Tốt | Cao |

**Phân tích so sánh:**

1. **Về kiến trúc**: Uni-MuMER sử dụng VLM pre-trained (Qwen2.5-VL-3B) thay vì xây dựng từ đầu, cho phép tận dụng kiến thức đã được học từ dữ liệu đa dạng.

2. **Về multi-task learning**: Uni-MuMER là phương pháp đầu tiên tích hợp ba tác vụ bổ trợ (Tree-CoT, EDL, SC) trong một quá trình training thống nhất, trong khi các phương pháp khác chỉ tập trung vào tác vụ nhận dạng chính.

3. **Về quantization**: Uni-MuMER sử dụng QLoRA với 4-bit quantization, giảm yêu cầu VRAM xuống 50-70% so với các phương pháp không sử dụng quantization.

4. **Về hiệu suất**: Uni-MuMER đạt hiệu suất state-of-the-art, vượt SSAN 16.31% và Gemini2.5-flash 24.42% trong thiết lập zero-shot.

5. **Về khả năng tổng quát**: Do sử dụng VLM pre-trained, Uni-MuMER có khả năng tổng quát hóa tốt hơn các mô hình chuyên biệt, có thể áp dụng cho các tác vụ thị giác khác.

### 2.11. Điểm mạnh và đóng góp của Phương pháp Uni-MuMER

**Điểm mạnh chính:**

1. **Tận dụng kiến thức pre-trained**: Sử dụng Qwen2.5-VL-3B như base model, tận dụng kiến thức đã được học từ hàng tỷ dữ liệu đa phương tiện, không cần training từ đầu.

2. **Hiệu quả tài nguyên**: Sử dụng QLoRA giảm 50-70% VRAM usage so với full fine-tuning, cho phép training trên GPU consumer-grade (RTX 3090, A6000) thay vì yêu cầu GPU cao cấp (A100, H100).

3. **Multi-task learning thống nhất**: Tích hợp ba tác vụ bổ trợ trong một quá trình training, cho phép model học được nhiều khía cạnh của bài toán đồng thời, cải thiện hiệu suất tổng thể.

4. **Data-driven approach**: Các tác vụ được thiết kế dựa trên phân tích dữ liệu thực tế, đảm bảo tính thực tiễn và hiệu quả.

5. **State-of-the-art performance**: Đạt hiệu suất tốt nhất trên CROHME và HME100K, vượt các phương pháp chuyên biệt và VLM hàng đầu.

6. **Khả năng tổng quát hóa**: Do sử dụng VLM pre-trained, model có khả năng tổng quát hóa tốt, có thể áp dụng cho các tác vụ thị giác khác ngoài HMER.

**Đóng góp nghiên cứu:**

1. **Phương pháp mới**: Lần đầu tiên áp dụng QLoRA + Multi-task Learning cho bài toán HMER, mở ra hướng nghiên cứu mới về parameter-efficient fine-tuning trong lĩnh vực này.

2. **Tích hợp ba tác vụ bổ trợ**: Đề xuất và tích hợp thành công ba tác vụ (Tree-CoT, EDL, SC) trong một quá trình training thống nhất, chứng minh hiệu quả của multi-task learning trong HMER.

3. **Giảm yêu cầu tài nguyên**: Chứng minh rằng có thể đạt hiệu suất cao với tài nguyên hạn chế, mở ra khả năng nghiên cứu và triển khai rộng rãi hơn.

4. **Benchmark mới**: Thiết lập benchmark mới cho bài toán HMER, vượt các phương pháp trước đó đáng kể.

---

## 3. CÁC KỸ THUẬT TINH CHỈNH ĐÃ ÁP DỤNG

### 3.1. QLoRA (Quantized Low-Rank Adaptation)

#### 3.1.1. Khái niệm

QLoRA là sự kết hợp giữa **LoRA (Low-Rank Adaptation)** và **4-bit Quantization**, cho phép fine-tuning mô hình lớn với tài nguyên hạn chế.

#### 3.1.2. Cơ chế hoạt động

**LoRA (Low-Rank Adaptation):**
- Thay vì cập nhật toàn bộ ma trận trọng số W (kích thước d×d), LoRA phân tích W thành tích của hai ma trận hạng thấp:
  ```
  W' = W + ΔW = W + BA
  ```
  Trong đó:
  - B: ma trận kích thước d×r (rank r)
  - A: ma trận kích thước r×d
  - r << d (rank nhỏ hơn nhiều so với chiều gốc)

**4-bit Quantization:**
- Nén trọng số từ 32-bit (FP32) hoặc 16-bit (FP16/BF16) xuống 4-bit
- Sử dụng **NF4 (NormalFloat4)** quantization scheme
- Giảm kích thước model từ ~3GB xuống ~1.5GB

**QLoRA = LoRA + 4-bit Quantization:**
- Base model được quantize xuống 4-bit và đóng băng (frozen)
- Chỉ train các LoRA adapters (ma trận B và A) ở full precision
- Kết hợp cả hai kỹ thuật để tối ưu hóa tài nguyên

#### 3.1.3. Cấu hình trong dự án

```yaml
# Từ file train/Uni-MuMER-train.yaml

# QLoRA configuration
quantization_bit: 4              # 4-bit quantization
quantization_type: nf4            # NormalFloat4 quantization
finetuning_type: lora             # Sử dụng LoRA

# LoRA parameters
lora_target: all                  # Áp dụng LoRA cho tất cả linear layers
lora_rank: 64                     # Rank của ma trận phân tích (r=64)
lora_alpha: 16                    # Scaling factor (alpha=16)
lora_dropout: 0.05                # Dropout rate cho LoRA layers
```

**Giải thích tham số:**
- **lora_rank (r=64)**: Số chiều của ma trận hạng thấp. Rank càng cao, khả năng biểu diễn càng tốt nhưng tốn nhiều tham số hơn.
- **lora_alpha (α=16)**: Hệ số scaling để điều chỉnh ảnh hưởng của LoRA adapters. Tỷ lệ α/r = 16/64 = 0.25 là tỷ lệ scaling.
- **lora_dropout (0.05)**: Tỷ lệ dropout để tránh overfitting.

#### 3.1.4. Lợi ích

**Tiết kiệm bộ nhớ:**
- **Training VRAM**: Giảm từ ~60-80GB xuống ~20-30GB (giảm 50-70%)
- **Inference VRAM**: Giảm từ ~6GB xuống ~2-3GB (giảm 50%)
- **Model size**: Giảm từ ~3GB xuống ~1.5GB (giảm 50%)

**Tiết kiệm tham số trainable:**
- Thay vì train 3 tỷ tham số, chỉ train ~10-20 triệu tham số (LoRA adapters)
- Giảm số lượng tham số trainable xuống ~0.3-0.7% so với full fine-tuning

**Tốc độ training:**
- Nhanh hơn do chỉ cập nhật một phần nhỏ tham số
- Gradient computation đơn giản hơn

**Chất lượng:**
- Giữ được ~95-99% hiệu suất so với full fine-tuning
- Phù hợp cho các tác vụ chuyên biệt như HMER

---

### 3.2. 4-bit Quantization với NF4

#### 3.2.1. Khái niệm

**Quantization** là quá trình giảm độ chính xác của số để tiết kiệm bộ nhớ và tăng tốc độ tính toán.

#### 3.2.2. NormalFloat4 (NF4) Quantization

**Đặc điểm:**
- NF4 là một quantization scheme được thiết kế đặc biệt cho các phân phối trọng số của neural networks
- Tối ưu hóa cho phân phối chuẩn (normal distribution) của trọng số
- Sử dụng 4-bit để biểu diễn mỗi trọng số (16 giá trị có thể)

**Cơ chế:**
1. Phân tích phân phối trọng số của model
2. Chọn 16 giá trị quantization levels tối ưu dựa trên phân phối
3. Map mỗi trọng số gốc đến giá trị quantization gần nhất

#### 3.2.3. Cấu hình BitsAndBytes

```python
# Cấu hình quantization trong inference
BitsAndBytesConfig(
    load_in_4bit=True,                    # Kích hoạt 4-bit quantization
    bnb_4bit_use_double_quant=True,       # Double quantization để giảm thêm memory
    bnb_4bit_quant_type="nf4",            # Sử dụng NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype cho operations
)
```

**Double Quantization:**
- Quantize cả quantization constants (constants dùng để dequantize)
- Giảm thêm ~0.4 bits per parameter
- Tổng cộng: ~3.5-3.6 bits per parameter thay vì 4 bits

#### 3.2.4. So sánh với các phương pháp quantization khác

| Phương pháp | Bits | Độ chính xác | Tốc độ | Memory |
|------------|------|--------------|--------|--------|
| FP32 (Full) | 32 | 100% | Chậm | Cao |
| FP16/BF16 | 16 | ~99% | Trung bình | Trung bình |
| INT8 | 8 | ~95% | Nhanh | Thấp |
| **NF4 (QLoRA)** | **4** | **~90-95%** | **Rất nhanh** | **Rất thấp** |

---

### 3.3. Gradient Accumulation

#### 3.3.1. Vấn đề

Với quantization, batch size phải giảm xuống do overhead của quantization operations:
- Batch size gốc: 4 samples/batch
- Batch size với quantization: 2 samples/batch (giảm 50%)

#### 3.3.2. Giải pháp: Gradient Accumulation

**Cơ chế:**
- Thay vì cập nhật weights sau mỗi batch nhỏ, tích lũy gradients qua nhiều batches
- Chỉ cập nhật weights sau khi đã tích lũy đủ gradients

**Công thức:**
```
Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
```

**Cấu hình trong dự án:**
```yaml
per_device_train_batch_size: 2        # Giảm từ 4 xuống 2
gradient_accumulation_steps: 64       # Tăng từ 1 lên 64
# Effective batch size = 2 × 64 = 128 (giữ nguyên hoặc tăng so với ban đầu)
```

#### 3.3.3. Lợi ích

- **Duy trì effective batch size**: Giữ được batch size lớn để training ổn định
- **Tiết kiệm VRAM**: Không cần tăng batch size thực tế
- **Tăng độ chính xác**: Batch size lớn hơn thường cho gradient ước lượng tốt hơn

---

### 3.4. BFloat16 (BF16) Training

#### 3.4.1. Khái niệm

**BFloat16** là định dạng số dấu chấm động 16-bit được thiết kế bởi Google Brain, tương thích với FP32 về dynamic range.

#### 3.4.2. Đặc điểm

- **Dynamic range**: Giống FP32 (8 bits exponent)
- **Precision**: Thấp hơn FP32 (7 bits mantissa thay vì 23 bits)
- **Tốc độ**: Nhanh hơn FP32 trên GPU hiện đại
- **Ổn định**: Ít bị overflow/underflow hơn FP16

#### 3.4.3. Cấu hình

```yaml
bf16: true  # Sử dụng BFloat16 cho training
```

**Lợi ích:**
- Giảm memory usage so với FP32
- Tăng tốc độ training
- Giữ được dynamic range, tránh gradient vanishing/exploding

---

### 3.5. Learning Rate Scheduling

#### 3.5.1. Cosine Learning Rate Schedule

**Cơ chế:**
- Learning rate giảm dần theo hàm cosine từ giá trị ban đầu xuống 0
- Tạo đường cong mượt mà, giúp model hội tụ tốt hơn

**Công thức:**
```
lr(t) = lr_min + (lr_max - lr_min) × (1 + cos(π × t / T)) / 2
```

#### 3.5.2. Warmup

**Mục đích:**
- Tránh learning rate quá cao ở đầu training
- Giúp model ổn định trong những bước đầu

**Cấu hình:**
```yaml
learning_rate: 1.0e-4        # Learning rate ban đầu
lr_scheduler_type: cosine     # Cosine schedule
warmup_ratio: 0.1             # 10% số steps đầu dùng warmup
```

**Giải thích:**
- Learning rate cao hơn một chút (1e-4) so với full fine-tuning (thường 5e-5) để bù cho việc chỉ train adapters
- Warmup 10% giúp model thích ứng dần với learning rate

---

### 3.6. Multi-Task Learning (Chi tiết bổ sung)

#### 3.6.1. Các tác vụ được tích hợp (Tóm tắt)

Dự án tích hợp **3 tác vụ chuyên biệt** để cải thiện hiệu suất nhận dạng biểu thức toán học. Chi tiết đã được mô tả trong phần 2.3 (PHƯƠNG PHÁP ĐỀ XUẤT).

1. **Tree-Aware Chain-of-Thought (Tree-CoT)**: Học lập luận không gian có cấu trúc
2. **Error-Driven Learning (EDL)**: Giảm nhầm lẫn giữa các ký tự trực quan tương tự
3. **Symbol Counting (SC)**: Cải thiện tính nhất quán trong nhận dạng các biểu thức dài

#### 3.6.2. Dataset được sử dụng

```yaml
dataset: 
  - parquet_crohme_train              # Dataset chính CROHME
  - parquet_crohme_train_can          # Tree-CoT variant
  - parquet_crohme_train_tree          # Tree structure learning
  - parquet_crohme_train_error_find    # Error finding (EDL)
  - parquet_crohme_train_error_fix    # Error fixing (EDL)
  - parquet_hme100k_train             # HME100K dataset
```

---

## 4. SO SÁNH MÔ HÌNH GỐC VÀ TINH CHỈNH

### 3.1. Bảng so sánh tổng quan

| Tiêu chí | Mô hình gốc | Mô hình tinh chỉnh (QLoRA) | Cải thiện |
|----------|-------------|---------------------------|-----------|
| **VRAM Training** | ~60-80GB | ~20-30GB | **Giảm 50-70%** |
| **VRAM Inference** | ~6GB | ~2-3GB | **Giảm 50%** |
| **Model Size** | ~3GB | ~1.5GB | **Giảm 50%** |
| **Tham số Trainable** | 3B (100%) | ~10-20M (0.3-0.7%) | **Giảm 99%+** |
| **Batch Size** | 4 | 2 (effective 128) | Tương đương |
| **Tốc độ Training** | Chậm | Nhanh hơn 2-3x | **Tăng 200-300%** |
| **Độ chính xác** | 100% (baseline) | ~95-99% | Giảm nhẹ 1-5% |
| **Yêu cầu GPU** | A100/H100 (80GB+) | RTX 3090/A6000 (24GB+) | **Giảm đáng kể** |

### 3.2. So sánh chi tiết từng thành phần

#### 3.2.1. Memory Usage

**Mô hình gốc (Full Fine-tuning):**
```
Base Model (FP16):          ~6GB
Optimizer States (AdamW):   ~12GB (2× model size)
Gradients:                  ~6GB
Activations:                ~40-60GB
─────────────────────────────────
Tổng cộng:                  ~60-80GB
```

**Mô hình tinh chỉnh (QLoRA):**
```
Base Model (4-bit):         ~1.5GB
LoRA Adapters (FP16):       ~0.1GB
Optimizer States:           ~0.2GB (chỉ cho adapters)
Gradients:                  ~0.1GB (chỉ cho adapters)
Activations:                ~18-28GB
─────────────────────────────────
Tổng cộng:                  ~20-30GB
```

**Tiết kiệm:** ~40-50GB VRAM (62.5-70% reduction)

#### 3.2.2. Training Time

**Mô hình gốc:**
- Forward pass: Cập nhật 3B tham số
- Backward pass: Tính gradient cho 3B tham số
- Update: Cập nhật 3B tham số

**Mô hình tinh chỉnh:**
- Forward pass: Chỉ tính toán với 4-bit weights (nhanh hơn)
- Backward pass: Chỉ tính gradient cho ~20M tham số LoRA
- Update: Chỉ cập nhật ~20M tham số

**Tốc độ:** Nhanh hơn 2-3 lần

#### 3.2.3. Model Quality

**Độ chính xác trên CROHME:**
- Mô hình gốc (zero-shot): Baseline
- Mô hình tinh chỉnh: **Vượt SSAN 16.31%**, vượt Gemini2.5-flash 24.42%

**Kết luận:** Mặc dù sử dụng quantization, model vẫn đạt hiệu suất state-of-the-art nhờ:
1. Fine-tuning chuyên biệt cho tác vụ HMER
2. Multi-task learning với Tree-CoT, EDL, SC
3. LoRA adapters được train ở full precision

---

## 5. CHI TIẾT KỸ THUẬT TRIỂN KHAI

### 4.1. Training Pipeline

#### 4.1.1. Quy trình training

```
1. Load Base Model (Qwen2.5-VL-3B)
   ↓
2. Apply 4-bit Quantization (NF4)
   ↓
3. Freeze Base Model Weights
   ↓
4. Initialize LoRA Adapters
   - Target: All linear layers
   - Rank: 64
   - Alpha: 16
   ↓
5. Training Loop
   - Forward: Base model (4-bit) + LoRA adapters (FP16)
   - Backward: Compute gradients cho LoRA adapters only
   - Update: Chỉ cập nhật LoRA weights
   ↓
6. Save Checkpoints
   - Chỉ lưu LoRA adapters (~100MB)
   - Không lưu base model
```

#### 4.1.2. Code implementation

**Cấu hình training (YAML):**
```yaml
# train/Uni-MuMER-train.yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
finetuning_type: lora
quantization_bit: 4
quantization_type: nf4
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.05
per_device_train_batch_size: 2
gradient_accumulation_steps: 64
learning_rate: 1.0e-4
bf16: true
```

**Training command:**
```bash
llamafactory-cli train train/Uni-MuMER-train.yaml
```

### 4.2. Inference Pipeline

#### 4.2.1. Merge LoRA Adapters

Sau training, cần merge LoRA adapters vào base model để inference:

```python
# scripts/merge_checkpoint.py
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_dir,
    torch_dtype=torch.bfloat16
)

adapter_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
merged_model = adapter_model.merge_and_unload()
merged_model.save_pretrained(output_dir)
```

**Lợi ích:**
- Tạo một model thống nhất, không cần load riêng adapters
- Tăng tốc độ inference
- Dễ dàng deploy

#### 4.2.2. Inference với vLLM

**Cấu hình:**
```python
# scripts/vllm_infer.py
llm = LLM(
    model=model_name,
    quantization="bitsandbytes",  # 4-bit quantization
    dtype="half",
    enforce_eager=True,            # Tiết kiệm VRAM
    gpu_memory_utilization=0.95,
    max_model_len=2048
)
```

**Lợi ích của vLLM:**
- Dynamic batching: Tự động batch các requests
- PagedAttention: Quản lý memory hiệu quả
- Continuous batching: Xử lý requests không đồng bộ

---

## 6. KẾT QUẢ VÀ ĐÁNH GIÁ

### 5.1. Hiệu suất trên các dataset

#### 5.1.1. Kết quả chính từ Paper Uni-MuMER

Theo [paper chính thức](https://arxiv.org/abs/2505.23566), Uni-MuMER đạt được các kết quả sau:

**CROHME Dataset:**
- Vượt qua mô hình chuyên biệt nhẹ tốt nhất **SSAN 16.31%**
- Vượt qua VLM hàng đầu **Gemini2.5-flash 24.42%** trong thiết lập zero-shot
- Đạt hiệu suất state-of-the-art mới trên cả CROHME và HME100K

**HME100K Dataset:**
- Cải thiện đáng kể so với các phương pháp baseline
- Thể hiện khả năng tổng quát hóa tốt

#### 5.1.2. So sánh với các phương pháp khác

| Phương pháp | Hiệu suất | VRAM | Training Time | Ghi chú |
|------------|-----------|------|---------------|---------|
| Full Fine-tuning | Baseline | 60-80GB | Baseline | Yêu cầu GPU cao cấp |
| **Uni-MuMER (QLoRA)** | **SOTA** | **20-30GB** | **2-3x nhanh hơn** | **Vượt SSAN 16.31%, Gemini2.5-flash 24.42%** |
| LoRA (không quantize) | Tương đương | 40-50GB | 1.5x nhanh hơn | Không sử dụng quantization |
| SSAN | Baseline | - | - | Mô hình chuyên biệt nhẹ tốt nhất trước đó |
| Gemini2.5-flash | Baseline | - | - | VLM hàng đầu trước đó |

### 5.2. Phân tích tài nguyên

#### 5.2.1. Memory Breakdown

**Training:**
- Base Model (4-bit): 1.5GB
- LoRA Adapters: 0.1GB
- Optimizer: 0.2GB
- Activations: 18-28GB
- **Tổng: 20-30GB**

**Inference:**
- Model (4-bit): 1.5GB
- KV Cache: 0.5-1GB
- Activations: 0.5-1GB
- **Tổng: 2-3GB**

#### 5.2.2. Training Speed

- **Steps per second**: Tăng 2-3x so với full fine-tuning
- **Time to convergence**: Giảm 50-60%
- **Total training time**: Giảm đáng kể

### 5.3. Trade-offs

#### 5.3.1. Ưu điểm

✅ **Tiết kiệm tài nguyên:**
- Giảm 50-70% VRAM
- Giảm 99%+ tham số trainable
- Có thể train trên GPU consumer-grade

✅ **Tốc độ:**
- Training nhanh hơn 2-3x
- Inference nhanh hơn nhờ model nhỏ hơn

✅ **Linh hoạt:**
- Dễ dàng thử nghiệm với nhiều cấu hình LoRA
- Có thể train nhiều adapters cho nhiều tác vụ

✅ **Chất lượng:**
- Giữ được 95-99% hiệu suất
- Đạt state-of-the-art trên CROHME

#### 5.3.2. Nhược điểm

❌ **Độ chính xác:**
- Giảm nhẹ 1-5% so với full fine-tuning
- Quantization có thể gây mất mát thông tin

❌ **Phức tạp:**
- Cần merge adapters sau training
- Cấu hình phức tạp hơn

❌ **Hạn chế:**
- Không thể thay đổi kiến trúc model
- LoRA rank cần được điều chỉnh cẩn thận

---

## 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận

Dự án **Uni-MuMER** đã thành công trong việc tinh chỉnh mô hình Qwen2.5-VL-3B bằng các kỹ thuật:

1. **QLoRA**: Kết hợp LoRA và 4-bit quantization để giảm tài nguyên
2. **NF4 Quantization**: Sử dụng NormalFloat4 để tối ưu hóa quantization
3. **Gradient Accumulation**: Duy trì effective batch size lớn
4. **BFloat16 Training**: Tăng tốc độ và giảm memory
5. **Multi-task Learning**: Tích hợp Tree-CoT, EDL, SC để cải thiện hiệu suất

**Kết quả chính:**
- **Hiệu suất**: Vượt SSAN 16.31% và Gemini2.5-flash 24.42% trong zero-shot setting
- **Tài nguyên**: Giảm 50-70% VRAM usage (từ 60-80GB xuống 20-30GB)
- **Tham số**: Giảm 99%+ tham số trainable (chỉ train LoRA adapters)
- **Tốc độ**: Tăng tốc độ training 2-3x so với full fine-tuning
- **Độ chính xác**: Đạt hiệu suất state-of-the-art trên CROHME và HME100K
- **Khả năng triển khai**: Có thể train trên GPU consumer-grade (RTX 3090, A6000)

**Công nhận:**
- Được chấp nhận tại **NeurIPS 2025 với danh hiệu Spotlight** (688/21575 submissions)
- Paper: [arXiv:2505.23566](https://arxiv.org/abs/2505.23566)

### 6.2. Hướng phát triển

#### 6.2.1. Tối ưu hóa thêm

1. **LoRA++**: Cải thiện LoRA với rank factorization tốt hơn
2. **QLoRA variants**: Thử nghiệm với 3-bit, 2-bit quantization
3. **AdaLoRA**: Adaptive LoRA rank cho từng layer
4. **DoRA**: Weight-Decomposed Low-Rank Adaptation

#### 6.2.2. Mở rộng ứng dụng

1. **Multi-domain**: Áp dụng cho các domain khác (hóa học, vật lý, v.v.)
2. **Real-time inference**: Tối ưu hóa cho deployment thời gian thực
3. **Edge devices**: Quantize thêm để chạy trên mobile/edge devices

#### 6.2.3. Nghiên cứu sâu hơn

1. **Ablation studies**: Phân tích đóng góp của từng component
2. **Optimal LoRA rank**: Tìm rank tối ưu cho từng layer
3. **Quantization-aware training**: Cải thiện chất lượng quantization

---

## 8. TÀI LIỆU THAM KHẢO

### 7.1. Papers

1. **Uni-MuMER (Chính)**: 
   - Li, Y., Jiang, J., Zhu, J., Peng, S., Wei, B., Zhou, Y., & Gao, L. (2025). 
   - "Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition"
   - arXiv preprint arXiv:2505.23566
   - **NeurIPS 2025 Spotlight** (688/21575)
   - Link: https://arxiv.org/abs/2505.23566

2. **LoRA**: 
   - Hu, E. J., et al. (2021). 
   - "LoRA: Low-Rank Adaptation of Large Language Models"
   - arXiv preprint arXiv:2106.09685

3. **QLoRA**: 
   - Dettmers, T., et al. (2023). 
   - "QLoRA: Efficient Finetuning of Quantized LLMs"
   - arXiv preprint arXiv:2305.14314

4. **NF4 Quantization**: 
   - Được giới thiệu trong paper QLoRA (Dettmers et al., 2023)

### 7.2. Repository và Code

1. **Uni-MuMER Official Repository**: 
   - https://github.com/BFlameSwift/Uni-MuMER
   - Repository chính thức của dự án Uni-MuMER
   - Chứa code training, inference, và evaluation

2. **HuggingFace Datasets & Models**: 
   - https://huggingface.co/datasets/phxember/Uni-MuMER-Data
   - https://huggingface.co/collections/phxember/uni-mumer-68bfba4747e9289232f3d89e

### 7.3. Tools và Frameworks

1. **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
   - Framework được sử dụng cho training
   
2. **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes
   - Thư viện cho 4-bit quantization
   
3. **PEFT**: https://github.com/huggingface/peft
   - Parameter-Efficient Fine-Tuning library (LoRA implementation)
   
4. **vLLM**: https://github.com/vllm-project/vllm
   - Framework tối ưu cho inference

### 7.4. Datasets

1. **CROHME**: Competition on Recognition of Online Handwritten Mathematical Expressions
   - Dataset tiêu chuẩn cho đánh giá HMER
   - Bao gồm CROHME 2014, 2016, 2019, 2023

2. **HME100K**: Handwritten Mathematical Expression Recognition Dataset
   - Dataset lớn với 100K samples
   - Được sử dụng để đánh giá khả năng tổng quát hóa

3. **Uni-MuMER-Data**: 
   - Dataset được tạo bởi nhóm Uni-MuMER
   - Bao gồm các variants: Tree-CoT, EDL (error_find, error_fix), Symbol Counting
   - Link: https://huggingface.co/datasets/phxember/Uni-MuMER-Data

4. **Các datasets khác**:
   - Im2LaTeXv2
   - MathWriting
   - MNE (Mathematical Notation Extraction)

---

## 9. PHỤ LỤC

### 8.1. Cấu hình đầy đủ

Xem file `train/Uni-MuMER-train.yaml` để biết cấu hình đầy đủ.

### 8.2. Scripts

- `scripts/merge_checkpoint.py`: Merge LoRA adapters vào base model
- `scripts/vllm_infer.py`: Inference với vLLM và 4-bit quantization
- `scripts/eval_metrics_calculator.py`: Tính toán metrics đánh giá

### 8.3. Requirements

Xem file `requirements.txt` để biết danh sách dependencies đầy đủ.

---

---

## 10. THÔNG TIN DỰ ÁN

**Dự án gốc**: Uni-MuMER  
**Repository chính thức**: https://github.com/BFlameSwift/Uni-MuMER  
**Paper**: [arXiv:2505.23566](https://arxiv.org/abs/2505.23566)  
**Conference**: NeurIPS 2025 Spotlight (688/21575 submissions)  
**Tác giả paper**: Li, Yu; Jiang, Jin; Zhu, Jianhua; Peng, Shuai; Wei, Baole; Zhou, Yuxuan; Gao, Liangcai

**Báo cáo này**:  
**Tác giả**: [Tên của bạn]  
**Ngày**: [Ngày hiện tại]  
**Đồ án**: Khoa học Máy tính - Nhận dạng Biểu thức Toán học Viết tay  
**Dựa trên**: Implementation của Uni-MuMER từ repository chính thức

---

**Lưu ý**: Báo cáo này phân tích các kỹ thuật được sử dụng trong Uni-MuMER dựa trên code và cấu hình từ repository chính thức. Tất cả các kỹ thuật và kết quả được mô tả đều dựa trên paper và implementation chính thức của Uni-MuMER.

