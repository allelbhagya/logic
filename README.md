# logic

https://rikiyat.github.io/blog/weight-space-programming/ - to play ideas on this blog

to feed weights bias into model -> logic gates classification

# test -1: trained neural nets on transformer
- what its doing? training classification transformer on predefined model arch for 3 class
- what its not doing? if we change model arch, it may fail simply

<img width="521" height="454" alt="image" src="https://github.com/user-attachments/assets/72c2d42d-8d82-4a9e-878c-36ee80fd9774" />
<img width="1190" height="390" alt="image" src="https://github.com/user-attachments/assets/986d6156-9f42-48ed-a0fb-11c1a0eb6c7d" />


##### run

```bash
git clone https://github.com/allelbhagya/logic
cd logic/logic_dataset
python generate.py
python train.py
```

## generate.py
- 100 x 3 class: XOR, AND, OR
generating weights bias for model/ 300 samples of neural network w&b

2 input neurons
2 hidden neurons
1 output neuron

- generate n (100) weights and bias for each logic gate, accept weights and bias which are learnt, details of model arch at base.ipynb

- save in format:

```json
  {
    "architecture": [
      2,
      2,
      1
    ],
    "activation": "sigmoid",
    "weights": {
      "w1": [
        [
          1.6255753529743724,
          2.788180734279431
        ],
        [
          -5.561475614128438,
          -5.449756455360834
        ]
      ],
      "w2": [
        [
          3.776442017290667,
          -14.881403648745616
        ]
      ]
    },
    "biases": {
      "b1": [
        0.3460753495938545,
        7.917480329433189
      ],
      "b2": [
        3.0769960146864532
      ]
    },
    "gate": "AND",
    "predictions": [
      6.859268415450712e-05,
      0.0008908067584731701,
      0.0007453468871576859,
      0.9979256211482582
    ]
  }
```

## train.py

#### data encoder

- using 3 tokens per sample / 3 tokens per network

each sample dim: [3 x 4]

```markdown

Train, feed-forward: 

Input sample: 3 x 4
Dense (16) -> 3 x 16
MHA (2 attention heads, key_dim = 8) -> (3 x 8) + (3 x 8) concat -> 3 x 16
LayerNorm -> 3 x 16
FF MLP -> 3 x 16 -> 3 x 32 -> 3 x 16 -> 3 x 16
Global avg pool: 3 x 16 -> 16

Output class:
Dense(3) + softmax
16 -> 3
```
