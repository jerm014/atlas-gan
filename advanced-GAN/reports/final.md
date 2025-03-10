**Introduction:**

In this project, we set out to build and improve a Generative Adversarial Network for generating images similar to those in the CIFAR-10 
dataset. We leveraged TensorFlow and Keras to implement the models, and we carried forward insights from the prior DCGAN MNIST project to 
tackle the challenges of CIFAR-10. The project was structured into multiple experiments – a baseline implementation, an improved GAN, and 
finally hyperparameter tuning – each documented and tracked meticulously. This report summarizes our approach, findings, and key takeaways.

**Methodology:**

We began with a baseline DCGAN architecture, which included a convolutional generator and discriminator following well-established guidelines 
([DCGAN Explained](https://paperswithcode.com/method/dcgan#:~:text=,the%20discriminator%20for%20all%20layer)). The CIFAR-10 images (32×32 color)
were normalized to [-1,1] to work with a tanh-activated generator output 
(also, [Data Preprocessing Techniques for CIFAR-10](https://www.restack.io/p/data-preprocessing-answer-cifar-10-techniques-cat-ai#:~:text=In%20this%20equation%2C%20each%20image,of%20neural%20networks%20during%20training)).
We trained the networks adversarially, alternating between discriminator and generator updates, and used techniques like label smoothing to 
encourage stable convergence. After establishing the baseline, we refined the architecture (Experiment 1) by just implementing Wasserstien 
with GP. In Experiment 2, we systematically varied hyperparameters such as learning rate and batch size to observe effects on performance. 
Then we went back and integrated Weights & Biases to log metrics and images, enabling a comparative analysis across runs.

**Results:**

The baseline GAN utterly failed in learning to generate images that captured the general color distributions of CIFAR-10. The advanced WGAN
architecture yielded a dumpster fire. The images were mostly useless noise. The hyperparameter experiments were also unhelpful.

**Discussion:**

The progression from baseline to advanced GAN was apparently poorly executed due to sleep deprivation and ultimately giving up. Small and 
large architectural changes (like an extra layer or more filters or a completely different approach) resulted in meaningless images that 
would make Monet blush.

**Conclusion:**

My [W]GAN on CIFAR-10 project resulted in a generative model that can not produce images mimicking the CIFAR-10 classes. Through a series 
of experiments, I completely failed to improve the quality of the generated images.

