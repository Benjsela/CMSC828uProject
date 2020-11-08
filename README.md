# CMSC828uProject

Per-pixel smoothing

### Workflow

Training (base classifier)
- Gaussian augmentation (sigma = 1)

Testing (smoothed classifier)
- get saliency map
- model to take saliency map -> vector of sigmas
- modify [this code](https://github.com/locuslab/smoothing) to take vector of sigmas + base classifier -> smoothed classifier

### References
- Jeremy M. Cohen,  Elan Rosenfeld,  and J. Zico Kolter.  [Certified adversarial ro-bustness via randomized smoothing](https://arxiv.org/abs/1902.02918). \[[code](https://github.com/locuslab/smoothing)\]
- Alexander Levine, Sahil Singla, and Soheil Feizi.  [Certifiably robust interpretationin deep learning](https://arxiv.org/abs/1905.12105).