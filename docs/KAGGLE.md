# Chạy EnsembleSVR-XACLE bằng Kaggle CLI

## Chạy dual T4 và lưu feature

Lệnh `train` hiện chạy hai tiến trình CUDA độc lập trên `cuda:0` và `cuda:1` cho từng model và từng split. Mỗi tiến trình xử lý một shard liên tiếp, không chồng lặp; kết quả được ghép theo chỉ số gốc. M2D, MS-CLAP, LAION-CLAP và MGA-CLAP dùng batch 8 **trên mỗi GPU**; Whisper Large V2 dùng batch 1 **trên mỗi GPU** để tránh CUDA OOM đã xảy ra. Whisper vẫn ở chế độ eval, không dùng FP16 hoặc thay model. Do tokenizer padding theo batch và pooling gốc, embedding text Whisper có thể khác với lần chạy batch 8 cũ.

Feature cuối cùng nằm tại `outputs/features/{train,validation,test}/{m2d_clap,ms_clap,laion_clap,mga_clap,whisper}.npz`. Split test dùng `meta_data/test_average.csv` và audio trong `wav/test/`; chương trình kiểm tra mỗi WAV test có đúng một hàng trong CSV trung bình. Mỗi file feature chứa audio/text embedding, cosine, angular, L1, L2, nhãn, tên WAV và chỉ số gốc. `outputs/features/manifest.json` ghi fingerprint, SHA256, phiên bản phần mềm và trạng thái hoàn tất; cache sai schema, thứ tự, kích thước, dtype hoặc hash bị từ chối. L1/L2 được tính **sau khi ghép toàn bộ split**, đúng phép chuẩn hóa theo max trong mã gốc. Có thể gắn Dataset feature đã tạo ở `/kaggle/input/datasets/qnv8071/xacle-extracted-features`; job sẽ tự dùng cache hợp lệ.

Sau khi feature hoàn tất, cùng một job gọi `fit_from_features()` để chạy SVR A/B, TNC ensemble, đánh giá và lưu artifact ở `outputs/version_config_kaggle/`. Sau đó job áp dụng đúng preprocessing, SVR, trọng số ensemble và clipping gốc lên feature test để tạo `test_predictions.csv`, `inference_result_for_test.csv` và `test_metrics.json` (SRCC, LCC, KTAU, MSE, N). `outputs/metrics/summary.json` gồm metric train, validation và test. Pipeline không trích feature lần thứ hai. `outputs/execution_report.json` ghi trạng thái và đường dẫn; `outputs/execution.log` ghi mốc thời gian và bộ nhớ GPU của hai worker.

Không có Kaggle API Secret được cấu hình cho job, nên job **không tự công bố Dataset**. Các file feature trước hết được lưu trong runtime, rồi trở thành Notebook Outputs của phiên bản thành công. Sau khi job hoàn tất, mở trang Notebook `https://www.kaggle.com/code/qnv8071/xacle-ensemble-training`, vào **Output → New Dataset**, chọn feature output, đặt slug `qnv8071/xacle-extracted-features` và chọn **Private**. Xác nhận Dataset được tạo trước khi coi trạng thái công bố là hoàn tất. Kaggle cũng cho phép gắn Notebook Output trực tiếp làm input cho một job sau. Không tải feature về Ubuntu rồi upload lại trừ khi thao tác trực tiếp này không khả dụng.

## Chuẩn bị

Máy Ubuntu cần `python3` và Kaggle CLI đã đăng nhập. Không tải lại hai Dataset. Kernel riêng tư `qnv8071/xacle-ensemble-training` gắn `qnv8071/xacle-2027-data` và `qnv8071/xacle-weights`. Từ thư mục gốc repository:

```bash
bash scripts/xacle-kaggle validate
python3 -m unittest discover -s tests -p 'test_kaggle_workflow.py'
bash scripts/xacle-kaggle publish
```

`validate` chỉ kiểm tra cấu hình cục bộ. Cấu trúc CSV, WAV và checkpoint được kiểm tra trong job Kaggle. Cấu hình nghiên cứu gốc không bị ghi đè; `config_kaggle.json` dùng đường dẫn mount đã xác minh.
`publish` tạo kernel riêng tư bằng `kaggle kernels push --no-run`; không khởi chạy GPU.

## Preflight (chỉ chạy sau khi cho phép gửi GPU job)

```bash
bash scripts/xacle-kaggle preflight
bash scripts/xacle-kaggle status
bash scripts/xacle-kaggle logs --follow
bash scripts/xacle-kaggle download
```

Xem `outputs/preflight_report.json` và `outputs/failure.json` trong thư mục kết quả được in ra. Chỉ chạy training khi năm model đều báo `ok`, CUDA hoạt động, và schema 2027 khớp. Mỗi lệnh `preflight` gửi đúng một phiên bản kernel, không tự thử lại.

## Training (cần cho phép riêng sau preflight)

```bash
bash scripts/xacle-kaggle train
bash scripts/xacle-kaggle status
bash scripts/xacle-kaggle logs --follow
bash scripts/xacle-kaggle download
```

Lệnh download chỉ tải các tệp dưới `outputs/`, tạo thư mục có dấu thời gian dưới `results/` và không ghi đè kết quả cũ. Các mô hình dự kiến ở `outputs/version_config_kaggle/A/SVR_A.pkl` và `outputs/version_config_kaggle/B/SVR_B.pkl`, cùng Normalizer, StandardScaler, `metrics.json`, `test_metrics.json`, CSV dự đoán train/validation/test, `config.json`, `execution_metadata.json` và `dependency_versions.json`. Xác nhận đầy đủ tệp sau tải; trạng thái `COMPLETE` riêng lẻ chưa chứng minh training thành công.

## Thiết kế và khả năng tương thích

Kaggle CLI hiện chỉ gửi nội dung `code_file` đối với script kernel. Launcher tạo entrypoint tự chứa mã Python và cấu hình cục bộ; không cần đưa mã lên Dataset mới. Hai repository ngoài được clone tại SHA ghi trong `kaggle/sources.json`. MGA bỏ một wildcard import dùng cho training (`tools.utils`) để tránh các dependency không liên quan đến inference. `PortableM2D` đọc kiến trúc từ tên thư mục chứa checkpoint, vì vậy runner tạo symlink có tên tương ứng ở `/kaggle/working`; checkpoint gốc trong `/kaggle/input` vẫn nguyên vẹn.

Môi trường cách ly Python 3.11.13 dùng PyTorch/TorchVision/TorchAudio 2.6.0 CUDA 11.8 và các bản dependency trong `kaggle/requirements-kaggle.txt`. Cách này giữ NumPy 1.26.4 mà mã gốc nhắm tới; driver CUDA mới có thể chạy binary CUDA 11.8, nhưng cần xác nhận bằng preflight. MS-CLAP 2023, hai model LAION và Whisper large-v2 được tải qua cơ chế gốc của package/Hugging Face. `laion-clap.pt` trong Dataset chưa có bằng chứng tương thích với hai encoder Hugging Face nên chưa dùng. Internet phải được bật. Lần chạy đầu sẽ tải dependency và model; lỗi mạng được lưu vào `outputs/failure.json`.
Môi trường tạm và hai source clone được xóa sau khi job kết thúc để phần output Kaggle chỉ chứa artifact cần tải.

Batch size của bốn model đầu vẫn là 8 như `config_submission1.json`; Whisper dùng 1 trên mỗi GPU theo kết quả OOM của lần chạy trước. Preflight thử từng model với một mẫu, giải phóng bộ nhớ sau mỗi lần. Pipeline training giữ thứ tự feature, khoảng cách, Normalizer, StandardScaler, hai SVR, tối ưu TNC, chuẩn hóa và clipping như mã gốc. `evaluate.py` tính Kendall tau theo cặp nên đánh giá tập lớn có thể chậm. Mã gốc cũng đưa audio 16 kHz vào processor LAION với tham số `sampling_rate=48000`; workflow giữ nguyên hành vi nghiên cứu. Không tự sửa phép tính đó.

## Xử lý lỗi

Nếu preflight báo thiếu WAV/CSV hoặc schema không phù hợp, dùng đường dẫn và cột trong `preflight_report.json` để cập nhật lớp tương thích trước khi training. Nếu model lỗi, xem traceback của model trong báo cáo. Nếu CUDA thiếu, job dừng; không chuyển về CPU. Nếu clone hay Hugging Face không tải được, xem `failure.json` và log, rồi chủ động gửi lại một job khi nguyên nhân đã được sửa. Không chạy đồng thời preflight và training trên cùng kernel slug vì một lần push mới tạo phiên bản mới.
