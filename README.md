# Deep Interest Evolution Network for Click-Through Rate Prediction with OpenVINO backend
https://arxiv.org/abs/1809.03672

This repo introduce how to run Deep Interest Evoluation Network (DIEN) with OpenVINO backend.

Please refer to orignal repo for details: https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN_TF2

## prepare data
### method 1
You can get the data from amazon website and process it using the script
```bash
sh prepare_data.sh
```
### method 2 (recommended)
Because getting and processing the data is time consuming，so we had processed it and upload it for you. You can unzip it to use directly.
```bash
sh prepare_dataset.sh
```
```
When you see the files below, you can do the next work. 
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info
```

## Setup Python Environment
```
pip install openvino openvino-dev[tensorflow]
```

## Convert original Tensorflow model to OpenVINO FP32 IR
```python
mo --input_meta_graph dnn_best_model_trained/ckpt_noshuffDIEN3.meta \
   --input "Inputs/mid_his_batch_ph[-1,-1],Inputs/cat_his_batch_ph[-1,-1],Inputs/uid_batch_ph[-1],Inputs/mid_batch_ph[-1],Inputs/cat_batch_ph[-1],Inputs/mask[-1,-1],Inputs/seq_len_ph[-1]" \
   --output dien/fcn/Softmax --model_name DIEN -o openvino/FP32 --compress_to_fp16=False
```

## Run the Benchmark

### Run the Benchmark with Tensorflow backend
```
./infer.sh tensorflow
```

### Run the Benchmark with OpenVINO backend using FP32 inference precision
```
./infer.sh openvino f32
```

### Run the Benchmark with OpenVINO backend using BF16 inference precision
```
./infer.sh openvino bf16
```
Please note, BF16 infer precision native support start from 4th Generation Intel® Xeon® Scalable Processors. Run BF16 on legacy xeon platform may lead to performance degratation. 
