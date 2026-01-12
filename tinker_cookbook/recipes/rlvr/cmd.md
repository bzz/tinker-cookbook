

## 1. v4a format-only reward

### Non-SFT
```sh
python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_applies_v4a.toml
```

_Temperature_: 1.0 (!)
Cost: $0.55
Reward: 0.4 -> 0.5

_TODO_: try lower temperature?

```sh
tinker --format json checkpoint list --run-id "7b544bf9-d8d0-5979-9ae9-43a3bd2f0f4a:train:0" \
  | jq -r '.checkpoints[].tinker_path'
```

Eval results: worse than SFT (no surprise)
 SFT: 0.46 0.63 0.36
RLVR: 0.56 0.36 0.15

Findings
 * indeed, many diffs are parsable now
 * but they seem longer and rarely add up to anything meaninfull



## Offline evaluation

```sh
time python -m tinker_cookbook.recipes.rlvr.offline_eval \
  --config tinker_cookbook/recipes/rlvr/configs/patch_exact_v4a.toml \
  max_eval_samples=32

# results were a bit different from llm-eval on shuffled daaset

python -m tinker_cookbook.recipes.rlvr.offline_eval \
  --config tinker_cookbook/recipes/rlvr/configs/patch_exact_v4a.toml \
  load_checkpoint_path=tinker://path/to/checkpoint \
  max_eval_samples=10
```


## 2. v4a format + exact match reward, w\ inline eval

### Non-SFT, t=0
_Temperature_: 0.0 (!) for train and eval
samples: validation 166 / test 32

Costs: $0.35, 2M
 * Start $3.3,   9.21M
 * End   $3.65, 11.61M

```
python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_exact_v4a.toml
```

### From SFT
_Temperature_: train 0.7 eval 0.0
samples: validation 166 / test 32

Costs:  $0.4, 2.6M
* Start $3.65 11.61M
* End   $4.04 14.25M


Continue from SFT checkpoing
```sh
python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_exact_v4a_sft.toml \
  load_checkpoint_path='tinker://83630c19-e5a0-58b2-aaf9-6f08f3bc4305:train:0/weights/final'
```

Resume previous training run, do another 1 epoch

Costs:
* Start $4.04 14.25M
* End   $4.39 16.54M

```sh
python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_exact_v4a_sft.toml \
  log_path='logs/rlvr-diff-xyz-v4a-Qwen-Qwen3-4B-Instruct-2507-gs4-bs8-lr1e-05-2026-01-08-12-46'
```

Doing that for N more ephochs



_TODO_: try other recomended hparams https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507#best-practices?
they are not exposed in CLIConfig / Trainer Config 
 wouldn't it be nice to have a torchtune style component configuratoin, so we don't need to declare & pass them in multiple places (CLI -> Builders -> Trainer)


## 3. same + punish context verbosity 
`PatchExactMatchMinimalDiffSmallContextEnv` -> `PatchExactMatchMinimalDiffWideGapEnv` in 12e

```sh
time python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_minimal_v4a_sft.toml \
  log_path='logs/rlvr-diff-xyz-v4a-Qwen-Qwen3-4B-Instruct-2507-gs4-bs8-lr1e-2026-01-10-minimal'
```

166 exmples x4
Cost
 1. $0.4, 2.6M tokens, 13min
 2. ... 11m
 3. $0.7, 4.6M tokens, 30min
 4. $0.35	2.3M tokens, 13min
 5.
 6-7-8
 9-10. 
 11. group_size=8 (x2 samples)
   start $9.88 52.62M
   end  $10.61 57.34M

 12. wider minimality gap -> `PatchExactMatchMinimalDiffWideGapEnv`

 ```
 time python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_minimal_v4a_sft.toml \
  log_path='logs/rlvr-diff-xyz-v4a-Qwen-Qwen3-4B-Instruct-2507-gs4-bs8-lr1e-2026-01-10-minimal' num_epochs=1 eval_every=10 env_class='tinker_cookbook.recipes.rlvr.patch_env:PatchExactMatchMinimalDiffWideGapEnv'
```

Findings
 * minimality penalty gap that doesn't produce meaningfull gradients
 * compressed advantage magnitute


## 4. Reward for similarity to ground truth v4a (no SFT)
`PatchHybridSimilarityEnv`

```
time python -m tinker_cookbook.recipes.rlvr.train \
  --config tinker_cookbook/recipes/rlvr/configs/patch_similarity_v4a.toml \
  log_path='logs/rlvr-diff-xyz-v4a-Qwen-Qwen3-4B-Instruct-2507-gs4-bs8-lr1e-2026-01-10-similarity' num_epochs=4
```

Cost
* train $1.35, 8.75M
* eval table \w 4 rows, 100 examples
  4 x 0,133*$0,07+0,032*$0,22 = 4x$0.02; 
  actual bill: $0.05


Observations
 * minimality is not rewarded ATM 
 
 TODO: bonus on top similarity to v4a from gpt-oss


 ## Evaluation

1. Get trained model path checkpoint
```
# Get a checkpoint for a specific run
tinker --format json checkpoint list --run-id "aa7b7143-ec73-5dc5-b71c-6182b8570ac3:train:0" \
  | jq -r '.checkpoints[].tinker_path'

```

2. Add it to .gin file 

```
include 'experiments/diff-xyz/diff_generation/v4a/base_instruct_latest.gin'
import llm_eval.common.model

Experiment.description = """REST API: RLVR over SFT for Qwen3-4b-Instruct using Tinker API \w full v4a instructions & application-only reward"""

Experiment.model_cls = @OpenaiCompatibleApiModel
Experiment.model_hparam = {"temperature": 0, "max_tokens": 4096}

# Tinker https://tinker-docs.thinkingmachines.ai/compatible-apis/openai
OpenaiCompatibleApiModel.base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
OpenaiCompatibleApiModel.model_name = "tinker://<run-id>/sampler_weights/Qwen3-4B-Inst_v4a-exact_final"
```


3. Launch evaluation

```
 time OPENAI_API_KEY="${TINKER_API_KEY}" python src/llm_eval/main.py --task=diff_gen_v4a_easy_sft --experiment experiments/diff-xyz/diff_generation/v4a/qwen3-4b-instruct-tinker-rl-sft-exact_instruct.gin --limit 100 --use_coroutines --rpm_limit 60
 ```