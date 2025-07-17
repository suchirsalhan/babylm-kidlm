```
git clone https://github.com/suchirsalhan/babylm-kidlm
bash setup.sh
pip install -e .  # From the repo root with the pyproject.toml
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
