### Benchmarks

The paper reports results on **three benchmarks**:

- Graph Count
- BREC
- TSP

## 🚀 Key Results

| Domain | Benchmark | Result |
| :--- | :--- | :--- |
| **Algorithmic** | CLRS-30 | **96.64%** (SOTA), effectively solving graph & string algorithms. |
| **Optimization** | Non-Metric TSP | **88.3%** optimality on unseen sizes ($N=200$), vs 1.3% for Linkern heuristic. |
| **Expressiveness** | Substructure Count | Near-zero error (MAE **0.001**) on complex substructure counting. |

### Graph Count

The Graph Count benchmark and dataset construction follow:
https://github.com/subgraph23/homomorphism-expressivity

```bash
source .venv/bin/activate
cd example
python -m data.count.process_data
./count/run.sh
```

### BREC

The BREC benchmark and dataset construction follow:
https://github.com/GraphPKU/BREC

```bash
source .venv/bin/activate
cd example/data/BREC/raw
unzip BREC_data_all.zip

# Back to the example folder
cd ../../..

# Reproduce FloydNet results
python -m BREC.test_BREC

# Reproduce 3-Floyd results
python -m BREC.test_BREC --floyd_level 3
```

### TSP

Reproducing TSP at full scale is computationally heavy and involves large datasets. For convenience, we provide:

- A small demo dataset on Hugging Face:  
  https://huggingface.co/datasets/ocxlabs/FloydNet_TSP_demo
- Pretrained checkpoints for:
  - **Metric TSP** (Euclidean TSP, `euc`): https://huggingface.co/ocxlabs/FloydNet_TSP_euc
  - **Non-metric TSP** (Explicit TSP, `exp`): https://huggingface.co/ocxlabs/FloydNet_TSP_exp

This section describes **inference and evaluation** using the demo data and checkpoints.

#### Prepare demo data

Download the demo dataset as a `.zip`, unzip it, and place the extracted folder under `example/data/`.

#### Inference

Run inference in `--test_mode` using `torchrun` (the command below assumes **single-node, 8 GPUs**).  
Set `--subset` and make sure `--load_checkpoint` matches the subset.

```bash
source .venv/bin/activate
cd example

torchrun \
  --nproc_per_node=8 \
  -m TSP.run \
  --subset exp \
  --output_dir ./outputs/TSP_exp \
  --load_checkpoint path/to/TSP_exp/epoch_01000.pt \
  --test_mode \
  --split_factor 1 \
  --sample_count_per_case 10
```

#### Evaluation

After inference finishes, aggregate results with:

```bash
source .venv/bin/activate
cd example

python -m TSP.report ./outputs/TSP_exp
```

This saves CSV summaries (downsampled to 1 / 5 / 10 samples per instance) into the same `output_dir`.

#### Data generation

If you want to generate additional data (beyond the demo set) and train from scratch, prepare the raw `.npy` files as follows.

##### Metric TSP (Euclidean TSP, `euc`)

1. Randomly sample **N integer points** in 2D, $p_i = (x_i, y_i)$, and ensure **pairwise Euclidean distances ≤ 200**.
2. Solve the instance with a classic TSP solver (e.g., [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html)).
3. Reorder the points so that $p_0 \rightarrow p_1 \rightarrow ... \rightarrow p_{N-1}$ is the optimal tour.
4. Sample **T** instances for each **N**, stack them into a NumPy array of shape **`[T, N, 2]`** with dtype **`int8`**, and save grouped-by-N arrays as:
   - `data/TSP/euc/non-uni/raw/{N:03d}.npy`

##### Non-metric TSP (Explicit TSP, `exp`)

1. Randomly sample an **N×N symmetric distance matrix** with **maximum value ≤ 200**.
2. Solve with a classic TSP solver (e.g., [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html)).
3. Reorder rows/columns so that $0 \rightarrow 1 \rightarrow ... \rightarrow N-1$ is the optimal tour.
4. Sample **T** instances for each **N**, stack them into a NumPy array of shape **`[T, N, N]`** with dtype **`int8`**, and save grouped-by-N arrays as:
   - `data/TSP/exp/non-uni/raw/{N:03d}.npy`

#### Training from scratch

Recommended (paper-matching) training setup: 8 nodes × 8 GPUs = 64 GPUs total.

```bash
source .venv/bin/activate
cd example
torchrun \
  --master_addr <MASTER_ADDR> \
  --master_port <MASTER_PORT> \
  --nnodes 8 \
  --node_rank <NODE_RANK> \
  --nproc_per_node 8 \
  -m TSP.run \
  --subset exp \
  --output_dir ./outputs/TSP_exp \
  --wandb_name TSP_exp
```

---