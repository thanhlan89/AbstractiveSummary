# AbstractiveSummary

Phần này triển khai hướng 1 của bài tập lớn: xây dựng và huấn luyện mô hình Transformer encoder-decoder hoàn toàn từ đầu bằng PyTorch cho bài toán abstractive summarization.

## Thành phần đã tự cài

- Scaled Dot-Product Attention
- Multi-Head Attention
- Sinusoidal Positional Encoding
- Position-wise Feed Forward Network
- Encoder layer và Decoder layer
- Stacked Transformer Encoder/Decoder
- Causal mask cho decoder và padding mask cho input
- Huấn luyện teacher forcing, label smoothing, gradient clipping, Noam learning-rate schedule
- Đánh giá ROUGE-1, ROUGE-2, ROUGE-L

Code không dùng `nn.Transformer`, `nn.TransformerEncoder`, hoặc `nn.TransformerDecoder`.

## Chuẩn dữ liệu

Mặc định script đọc `data/train.tsv` và `data/valid.tsv` với header:

```tsv
text	summary
van ban dai...	tom tat...
```

Ngoài TSV, script cũng đọc được CSV hoặc JSONL. Nếu tên cột khác, truyền thêm `--source-field` và `--target-field`.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Train Transformer tự xây dựng

Chạy cấu hình nhỏ để kiểm tra pipeline:

```bash
python -m src.train --train data/train.tsv --valid data/valid.tsv --epochs 1 --batch-size 8 --embed-dim 128 --hidden-dim 512
```

Chạy cấu hình gần base hơn nếu máy đủ GPU:

```bash
python -m src.train --train data/train.tsv --valid data/valid.tsv --epochs 20 --batch-size 16 --embed-dim 512 --num-heads 8 --encoder-layers 6 --decoder-layers 6 --hidden-dim 2048
```

Checkpoint, vocab và config được lưu trong thư mục `checkpoints/`.

## Sinh tóm tắt

```bash
python -m src.predict --checkpoint checkpoints/transformer_epoch_20.pt --text "Nhap van ban dai can tom tat o day."
```

## Ghi chú báo cáo

Mô hình thuộc nhóm sequence-to-sequence Transformer theo Vaswani et al. (2017). Encoder đọc văn bản nguồn, decoder sinh từng token tóm tắt theo cơ chế masked self-attention và encoder-decoder attention. Phần cải tiến so với cấu hình tối thiểu gồm label smoothing, lịch học Noam, gradient clipping và đánh giá ROUGE tự động trên tập validation.
