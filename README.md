# A Deep Reinforcement Learning Question Answering System for Complex Questions using Texts and Tables
Repository for the BRACIS'24 paper "A Deep Reinforcement Learning Question Answering System for Complex Questions using Texts and Tables".

For more information, please check the paper here: https://arxiv.org/abs/2407.04858 

## Abstract
```
This paper proposes a novel architecture to generate multi-hop answers to open domain questions that require information from texts and tables, using the Open Table-and-Text Question Answering dataset for validation and training. One of the most common ways to generate answers in this setting is to retrieve information sequentially, where a selected piece of data helps searching for the next piece. As different models can have distinct behaviors when called in this sequential information search, a challenge is how to select models at each step.  Our architecture employs reinforcement learning  to choose between different state-of-the-art tools sequentially until, in the end, a  desired answer is generated. This system achieved an F1-score of 19.03, comparable to iterative systems in the literature.
```

## Installation
To install all the necessary files, please refer to the dockerfile. 

```
docker build -t DRL__TT docker
```

## Acknowledgments
This research has been carried out with  support by *Itaú Unibanco S.A.* through the scholarship program  *Programa de Bolsas Itaú* (PBI); it is also supported in part by the *Coordenação de Aperfeiçoamento de Pessoal de Nível Superior* (CAPES), Finance Code 001, Brazil. The authors would like to thank the Center for Artificial Intelligence (C4AI-USP), with support by the São Paulo Research Foundation (FAPESP) under grant number 2019/07665-4 and by the IBM Corporation. Paulo Pirozelli is supported by the FAPESP grant 2019/26762-0. 

Any opinions, findings, and conclusions expressed in this manuscript are those of the authors and do not necessarily reflect the views, official policy or position of the Itaú-Unibanco, FAPESP, IBM, and CAPES.

## Cite this work
```
@misc{josé2024questionansweringtextstables,
      title={Question Answering with Texts and Tables through Deep Reinforcement Learning}, 
      author={Marcos M. José and Flávio N. Cação and Maria F. Ribeiro and Rafael M. Cheang and Paulo Pirozelli and Fabio G. Cozman},
      year={2024},
      eprint={2407.04858},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.04858}, 
}
```
