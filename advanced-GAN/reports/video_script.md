This is my quick walkthrough of the Advanced GAN with CIFAR-10
project.

Over the next few time units, I�ll give you an overview of what 
I did, how I did it, and what I learned. Which isn't much.

I started by implementing a baseline GAN using the DCGAN
architecture found in that paper from Radford et al. In fact, 
I made the model exactly like theirs. The results were nothing
short of craptastic.

Next, I moved to Experiment 1: Advanced GAN. I was really
interested in the Wasserstien thing, and I wasn't getting
anywhere with the other method, so I just dove in with both feet.
Would I get better results? No. Compared to the baseline images,
I will say, there was some learning. Not just colorful noise.

Then came Experiment 2: Hyperparameter Tuning. GANs can be
finicky, so I played with learning rates, batch sizes, and
latent dim space. I found that nothing I tried made anything any
better.

Throughout all these experiments, I used Weights & Biases
to keep track of everything. On the screen, you can see
W&B dashboard. I logged the generator and discriminator
losses for each run and saved sample generated images at
different epochs.

Key takeaways: GAN training is hard. I learned that GAN
training is hard. I learned that it feels terrible to fail so
miserably at a project. I did learn how to use Weights and
Biases. Have a great day and thank you for watching.
