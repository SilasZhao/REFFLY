# REFFLY

**REFFLY: Melody-Constrained Lyrics Editing Model**

[Paper](https://arxiv.org/abs/2409.00292), [Demo](https://anonymous-account-xxx.github.io/anonymous_demo/), [Website](https://arxiv.org/abs/2409.00292)

## Outlines

- [Environment Setup](https://github.com/SilasZhao/REFFLY/tree/main?tab=readme-ov-file#environment-setup)
- [Dataset](https://github.com/SilasZhao/REFFLY/tree/main?tab=readme-ov-file#dataset)
- [Inference](https://github.com/SilasZhao/REFFLY/tree/main?tab=readme-ov-file#generation-and-execution-of-visual-programs)

*This repo is still under construction
## TODO
- clean files to make them more readable
- update the evaluation dataset for prominent notes extraction
- update website
## Environment Setup

```bash
pip install -r requirements.txt
```

## Dataset

From (https://www.letras.com/mais-acessadas/), we collected ~3500 song lyrics using 
```
finetune_data/scrape_lyric.py
```
Then, we use 
```
finetune_data/process_data.py
```
to get the data at
```
finetune_data/generated_line_all.json
```

## Inference
use
```
score_analysis.py
```
to get music constraint from .mxl file.

use
```
generate_end_to_end.py
```
to revise the lyrics based on music constraints.

