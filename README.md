# DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation

<p align="center">
  <img src="./demo/teaser_final.png" alt="alt text" width="400">
</p>


Official pytorch code release of "[DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation](https://arxiv.org/abs/2408.06010)"

```
@misc{kim2024deeptalkdynamicemotionembedding,
      title={DEEPTalk: Dynamic Emotion Embedding for Probabilistic Speech-Driven 3D Face Animation}, 
      author={Jisoo Kim and Jungbin Cho and Joonho Park and Soonmin Hwang and Da Eun Kim and Geon Kim and Youngjae Yu},
      year={2024},
      eprint={2408.06010},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.06010}, 
}
```
## Settings

## Inference
```
cd DEEPTalk
python demo.py \
--DEMOTE_ckpt_path ./checkpoint/DEEPTalk/DEEPTalk.pth \
--DEE_ckpt_path ../DEE/checkpoint/DEE.tph \
--audio_path ../demo/sample_audio.wav

```
## Training

Certainly! Here's a revised version:

---

## Acknowledgements
We gratefully acknowledge the open-source projects that served as the foundation for our work:

- [EMOTE](https://github.com/radekd91/inferno)
- [learning2listen](https://github.com/evonneng/learning2listen)
- [PCME++](https://github.com/naver-ai/pcmepp)

## License
This code is released under the MIT License.

Please note that our project relies on various other libraries, including FLAME, PyTorch3D, and Spectre, as well as several datasets.
