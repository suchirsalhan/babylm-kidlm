```
git clone https://github.com/suchirsalhan/babylm-kidlm
python3.9 -m venv venvs/demo; source venvs/demo/bin/activate
bash setup.sh
pip install -e .  # From the repo root with the pyproject.toml
```

Command for Sam to Run
```
poetry run train --model_type opt \
  --seq_len 1024 \
  --dataset Talking-Babies/sam-training-preshuffled
```

```
poetry run train--model_type opt \
  --seq_len 1024 \
  --dataset Talking-Babies/opt-kidlm-1024-preshuffled \
  --accumulation_steps 1 \
  --use_deepspeed
```

```
poetry run train--model_type opt \
  --seq_len 1024 \
  --dataset Talking-Babies/opt-babylm-1024-preshuffled \
  --accumulation_steps 1 \
  --use_deepspeed
```

For Cambridge HPC

```
 sbatch launch_slurm.wilkes3 --model_type opt \
  --seq_len 1024 \
  --dataset Talking-Babies/opt-kidlm-1024-preshuffled \
  --accumulation_steps 1 \
  --use_deepspeed
```

```
 sbatch launch_slurm.wilkes3 --model_type opt \
  --seq_len 1024 \
  --dataset Talking-Babies/opt-babylm-1024-preshuffled \
  --accumulation_steps 1 \
  --use_deepspeed
```
