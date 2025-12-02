# slimStreamQ - Clean and efficient implementation of Streaming Q($\lambda$)

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![jax_badge][jax_badge_link]
![Static Badge](https://img.shields.io/badge/lines%20of%20code-644-green)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Paper 👉[📄](https://arxiv.org/pdf/2410.14606) | Original code 👉[👨‍💻](https://github.com/mohmdelsayed/streaming-drl) (in Pytorch)

## User installation
CPU installation:
```bash
python3 -m venv env_cpu
source env_cpu/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```
GPU installation if needed:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```

## Running experiments
To train a Stream Q($\lambda$) agent on Breakout on your local system, run:\
`
launch_job/atari/launch.sh
`

- To see the stage of training, you can check the logs in `experiments/atari/logs/test_Breakout/qlambda`
- The models and episodic returns are stored in `experiments/atari/exp_output/test_Breakout/qlambda`



[jax_badge_link]: https://tinyurl.com/5n8m53cy