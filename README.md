# **DEEPFILL-V2 DEMONSTRATION**

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vrindaprabhu/deepfillv2_colab/blob/main/DeepFillv2_Colab.ipynb)

Colab code for image inpainting. The repository is based out of [DeepFillv2 Pytorch](https://github.com/csqiangwen/DeepFillv2_Pytorch) awesome implementation. 
Link to the [Original Paper](https://arxiv.org/abs/1806.03589)


```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}

```

Make sure to change `runtime` in colab to GPU for quick results!

**NOTE**

- The inpainting is being done after resizing the image to 512x512. This can be changed in the RESIZE_TO parameter in the _config.py_ file.
- An example of an image and corresponding mask is present in the _example_ folder of the repo. The notebook also allows generation of random masks. The uploaded image and the masks will be present in the _input_ folder.
- This repo has not been tested for training. Inference works fine.

Check it out and let me know if any issues!

