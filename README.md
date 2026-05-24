# AbstractiveSummary

Repo này triển khai phần 1 của bài tập lớn: xây dựng và huấn luyện mô hình Transformer encoder-decoder hoàn toàn từ đầu bằng PyTorch cho bài toán tóm tắt văn bản abstractive.

## Phạm vi

Code tự cài đặt các thành phần chính của Transformer theo bài báo "Attention is All You Need":

- Scaled Dot-Product Attention
- Multi-Head Attention
- Sinusoidal Positional Encoding
- Position-wise Feed Forward Network
- Encoder layer và Decoder layer
- Stacked Transformer Encoder/Decoder
- Padding mask cho encoder và causal mask cho decoder
- Huấn luyện teacher forcing, label smoothing, gradient clipping, Noam learning-rate schedule
- Đánh giá ROUGE-1, ROUGE-2, ROUGE-L trên tập validation

Code không dùng `nn.Transformer`, `nn.TransformerEncoder`, hoặc `nn.TransformerDecoder`.

## Dữ liệu

Mặc định repo dùng 2 file parquet đang có trong thư mục `data/`:

- `data/train-00000-of-00001.parquet`
- `data/valid-00000-of-00001.parquet`

Loader cũng hỗ trợ TSV, CSV và JSONL. Nếu tên cột trong dữ liệu không phải `text` và `summary`, truyền thêm:

```bash
--source-field ten_cot_van_ban --target-field ten_cot_tom_tat
```

Nếu không truyền, loader sẽ thử tự nhận các tên cột phổ biến như `article`, `document`, `content`, `summary`, `abstract`, `highlights`, `target`.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Train nhanh để kiểm tra pipeline

```bash
python -m src.train --train data/train-00000-of-00001.parquet --valid data/valid-00000-of-00001.parquet --epochs 1 --batch-size 8 --embed-dim 128 --hidden-dim 512 --encoder-layers 2 --decoder-layers 2 --max-src-len 256 --max-tgt-len 64
```

## Train cấu hình gần Transformer base

Chạy cấu hình này khi có GPU đủ bộ nhớ:

```bash
python -m src.train --train data/train-00000-of-00001.parquet --valid data/valid-00000-of-00001.parquet --epochs 20 --batch-size 16 --embed-dim 512 --num-heads 8 --encoder-layers 6 --decoder-layers 6 --hidden-dim 2048 --max-src-len 512 --max-tgt-len 128
```

Checkpoint, vocab và config được lưu trong thư mục `checkpoints/`.

## Sinh tóm tắt

```bash
python -m src.predict --checkpoint checkpoints/transformer_epoch_20.pt --text "Nhập văn bản dài cần tóm tắt ở đây."
```

## Gợi ý viết báo cáo

Mô hình thuộc nhóm sequence-to-sequence Transformer theo Vaswani et al. (2017). Encoder đọc văn bản nguồn, decoder sinh từng token tóm tắt bằng masked self-attention và encoder-decoder attention. Các cải tiến so với cấu hình tối thiểu gồm label smoothing, lịch học Noam, gradient clipping và đánh giá ROUGE tự động trên tập validation.
