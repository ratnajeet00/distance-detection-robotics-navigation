# results - quantization + fps (before vs after)

yolov8n @640px, measured on this machine (i7-14700F + RTX 5070), fps after warmup:

| variant          | runs on | size   | speed     |
|------------------|---------|--------|-----------|
| fp32 torch       | cpu     | 6.3mb  | 31.7 fps  |
| fp32 torch       | gpu     | 6.3mb  | 177.6 fps |
| fp32 onnx        | cpu     | 12.2mb | 39.6 fps  |
| fp32 onnx        | gpu     | 12.2mb | 223.2 fps |
| int8 onnx        | cpu     | 4.9mb  | 26.8 fps  |
| int8 onnx        | gpu     | 4.9mb  | 123.8 fps |
| fp16 onnx        | cpu     | 6.2mb  | 22.2 fps  |
| fp16 onnx        | gpu     | 6.2mb  | 196.5 fps |

takeaways, in simple words:

- quantizing works: **12.2mb -> 4.9mb (2.5x smaller)**, same objects detected with
  the same confidence (proof: `08_real_model_int8.jpg`)
- the gpu is ~6x faster than cpu for this model
- int8 is a bit slower than fp32 on this big desktop cpu - on small/edge boards
  int8 is usually the winner, so measure on the actual target device
- fp16 helps on gpu, not on cpu

what was tried for int8 (the short journey):

| approach                                   | result                          |
|--------------------------------------------|---------------------------------|
| qint8 activations (easy default)           | model dead, all scores 0        |
| quinit8 act + all ops quantized            | class scores -> 0               |
| same but without matmul                    | all confidences stuck at 0.5    |
| **conv-only + class head kept fp32**       | works, 2.5x smaller - shipped   |

the shipped recipe lives in `src/optimization/quantize.py` (`quantize_int8()`).
