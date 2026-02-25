
## Demo Video

https://raw.githubusercontent.com/wuzhirong520/VTR-VLM/refs/heads/master/demo_video/01262.mp4

## Installation

```bash
conda create -n vtr_venv python=3.11
conda activate vtr_venv

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.5.7
```

Patch Qwen2.5-VL in package transformers:
```bash
cp files_to_patch_tranformers/old/modeling_qwen2_5_vl.py vtr_venv/lib64/python3.11/site-packages/transformers/models/qwen2_5_vl/
```

Download datasets from huggingface: [LVBench](https://huggingface.co/datasets/zai-org/LVBench), [VideoEval-Pro](https://huggingface.co/datasets/TIGER-Lab/VideoEval-Pro), [MLVU](https://huggingface.co/datasets/sy1998/MLVU_dev), [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench), [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME). Then, modify "video_dir" and "anno_path" in "./configs/benchmarks".

Download models from huggingface: [LLaVA-Video-7B](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2), [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [PE-L/14](https://huggingface.co/facebook/PE-Core-L14-336), [PE-G/14](https://huggingface.co/facebook/PE-Core-G14-448). Then, modify "model_path" in "./configs/vlm_models" and "./configs/vtr_models".

## Demo Scripts
We provide four Python scripts to evaluate our method:

```bash
python demo_llava.py # run llava-video with our method
python demo_qwen2.5vl.py # run qwen2.5vl with our method
python demo_llava_with_pre_calc_emb.py # run llava-video with our method  (using pre-extracted embeddings)
python demo_qwen2.5vl_with_pre_calc_emb.py # run qwen2.5vl with our method (using pre-extracted embeddings)
```

Before execution, please update the model_path for both the VTR and VLM models within these scripts. Once configured, run any script. You will be prompted to interactively input the Video Path, Question, and Options.

Note that if you are testing videos from the five benchmarks (LVBench, VideoEval-Pro, MLVU, LongVideoBench, and VideoMME), we recommend using the provided pre-extracted embeddings (pre_calc_emb.zip).


## Embedding Extraction

You can directly use our pre-extracted embeddings located in **pre_calc_emb.zip**. Please ensure Git LFS is installed and enabled to download this file, then unzip it.

If you prefer to extract the embeddings yourself, please follow these steps:

- Modify "host" and "port" in "./configs/multinode/default.yaml" to set the distributer's ipv6 address and port.

- Modify "./configs/default_emb.yaml".

- Start the following code on main node:
```python
python eval/emb_distributer.py
```

- Start the following code on main node or other nodes:
```python
python eval/emb_runner.py
```

Embedding results are saved in "./logs/emb/XXXX".

## Benchmark Evaluation

- Modify "host" and "port" in "./configs/multinode/default.yaml" to set the distributer's ipv6 address and port.

- Modify "./configs/default.yaml".

- Start the following code on main node:
```python
python eval/vlm_distributer.py --config ./configs/default.yaml
```

- Start the following code on main node or other nodes:
```python
python eval/vlm_runner.py
```

Evaluation results are saved in "./logs/XXXX".

We give some example scripts to evaluate our performance.
```bash

python eval/vlm_distributer.py --config ./configs/examples/llava_lvbench.yaml
python eval/vlm_distributer.py --config ./configs/examples/llava_videoevalpro.yaml
python eval/vlm_distributer.py --config ./configs/examples/llava_mlvu.yaml
python eval/vlm_distributer.py --config ./configs/examples/llava_videomme.yaml
python eval/vlm_distributer.py --config ./configs/examples/llava_longvideobench.yaml

python eval/vlm_distributer.py --config ./configs/examples/qwen2.5vl_lvbench.yaml
python eval/vlm_distributer.py --config ./configs/examples/qwen2.5vl_videoevalpro.yaml
python eval/vlm_distributer.py --config ./configs/examples/qwen2.5vl_mlvu.yaml
python eval/vlm_distributer.py --config ./configs/examples/qwen2.5vl_videomme.yaml
python eval/vlm_distributer.py --config ./configs/examples/qwen2.5vl_longvideobench.yaml

```






