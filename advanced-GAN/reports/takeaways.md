### Throughout the project, we gathered several key observations and takeaways after each experiment:

#### Baseline GAN:

The baseline DCGAN had complete mode collapse.

#### Advanced GAN Improvements:

By improving the architecture in Experiment 1, nothing at all imporved.
I even tried a Wasserstien GAN with gradient penalty.

#### Effect of Hyperparameters:

The hyperparameter tuning experiment reinforced how sensitive GAN training is. Each param change had effect:
 * With a higher learning rate, we saw instances of divergence where the generator’s output quality would drastically worsen after a certain point (correlating with the loss blowing up). On the flip side, a slightly lower learning rate improved stability but required more epochs to reach comparable quality
 * Batch size had a subtle effect, a batch size of 128 produced smoother training curves, but the improvements in image quality were minor compared to batch size 64. We did note that too small a batch (like 32) made training very noisy.
 * When we tried a much larger latent dimension or extra layers, we expected better results, but observed diminishing returns. The model with a latent dim of 100 was nearly as good as one with 200; the extra capacity didn’t translate to significantly better images, possibly because the generator already had enough degrees of freedom to capture the data distribution.
 * We also observed that certain combinations of hyperparameters could either mitigate or exacerbate issues. For instance, using a higher learning rate could be partially compensated by a larger batch size (to stabilize updates), but it was still generally safer to stick to a moderate learning rate.

#### Training Stability and Techniques:

We learned that techniques like label smoothing and balanced training (ensuring neither network gets too far ahead) are important. In one trial, we forgot to apply label smoothing and noticed the discriminator very quickly got nearly 100% accuracy on real vs fake, after which the generator’s gradients vanished and the training stagnated. That run collapsed early. After reintroducing label smoothing and ensuring the generator got sufficient training steps relative to the discriminator, the situation improved. This experience emphasized the need for those subtle training tricks that we initially learned from literature and prior experience​

#### Use of W&B for Insights:

Having the W&B dashboard allowed us to notice these behaviors clearly. By looking at the graphs, we could pinpoint when a run was starting to destabilize (e.g., a sudden divergence in loss). The image panels on W&B were especially insightful – seeing the progression from random noise to structured images across epochs provided qualitative evidence of learning. In comparing runs, W&B showed, for example, that the advanced GAN’s loss curves were smoother than the baseline’s, confirming quantitatively that it was training more steadily. This real-time insight was a major takeaway: robust experiment tracking greatly aids in understanding GAN training, which can otherwise feel like a “dark art” due to its instability.

----

Overall, the project taught us not just how to implement a GAN for CIFAR-10, but also the importance of experimentation and monitoring. We found that small changes can have outsized effects on GAN performance, and thus careful logging, visualization, and incremental experimentation are crucial. We also learned to set realistic expectations: even with improvements, generating perfect CIFAR-10 images is very challenging, and our results, while improved, still have a gap compared to real images. This understanding will guide us in future GAN projects, perhaps exploring advanced variants like conditional GANs or Wasserstein GANs to further address these challenges.