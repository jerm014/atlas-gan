# DCGAN Architecture Variations Experiment Report

### points

* Provides an overview of what you did for the project, your reasoning behind decisions, and what you learned
* Walks through the Weights & Biases dashboard, explaining the key metrics, hyperparameters, and results from each experiment
* Highlights 2-3 main takeaways or observations from the experiments


### Experiment Overview

For this experiment, I modified the original DCGAN architecture to explore how architectural variations affect the model's performance and the quality of generated images. The main goal was to analyze the impact of:

Number of layers (I added intermediate layers)
Filter sizes (I tried using smaller 3×3 filters instead of 5×5)
Filter counts (I tried different channel progression)
Additional components in the networks (pointless)

#### Folder Structure

````
- dcgan/ - Root folder for the DCGAN project
    -  experiments/ - Contains MNIST dataset and any preprocessed versions and images and parameters used
         -   [experiment name]/ 
              -   images/  -  Contains images from the last run of this experiment
              -   hyperparamters.json  -  contains the hyperparameters for the experiment
              -   training_metrics.json  -  contains the training metrics for the experiment
    -  README.md - Overview of repo contents and documntation
    - 
````

#### Generator Modifications

| Component | Original Architecture | Modified Architecture |
|-----------|----------------------|----------------------|
| Initial Dense Layer | 7×7×256 (12,544 neurons) | 7×7×128 (6,272 neurons) |
| First Upsampling | Conv2DTranspose(128, 5×5) | Conv2DTranspose(64, 3×3) |
| Intermediate Layer | None | Added Conv2D(64, 3×3) layer |
| Second Upsampling | Conv2DTranspose(64, 5×5) | Conv2DTranspose(32, 3×3) |
| Refinement Layer | None | Added Conv2D(16, 3×3) layer |
| Output Layer | Conv2D(1, 5×5) | Conv2D(1, 3×3) |

#### Discriminator Modifications

| Component | Original Architecture | Modified Architecture |
|-----------|----------------------|----------------------|
| First Conv Layer | Conv2D(64, 5×5, stride=2) | Conv2D(32, 3×3, stride=2) |
| Intermediate Layer 1 | None | Added Conv2D(64, 3×3) |
| Second Conv Layer | Conv2D(128, 5×5, stride=2) | Conv2D(128, 3×3, stride=2) |
| Intermediate Layer 2 | None | Added Conv2D(256, 3×3) |
| Dense Layers | Flatten → Dense(1) | Flatten → Dense(128) → Dense(1) |

### Key Architectural Differences

1. **Deeper Networks**: Both the generator and discriminator have more layers in the modified architecture.
2. **Smaller Filters**: Switched from 5×5 convolutions to 3×3 convolutions, which are more computationally efficient while still capturing local features.
3. **Progressive Feature Refinement**: Added intermediate convolutional layers without upsampling/downsampling to refine features.
4. **Different Channel Progression**:
   - Original Generator: 256 → 128 → 64 → 1
   - Modified Generator: 128 → 64 → 64 → 32 → 16 → 1
   - Original Discriminator: 64 → 128 → 1
   - Modified Discriminator: 32 → 64 → 128 → 256 → 128 → 1

### Expected Impact of Modifications

**Deeper architecture with more layers**: 
   - Pros: fill this in when you figure it out, jerm.
   - Cons: Increased training time

**Smaller 3×3 convolution filters instead of 5×5**:
   - Pros: fill this in when you figure it out, jerm.
   - Cons: fill this in when you figure it out, jerm.

**Modified filter progression with additional refinement layers**:
   - Pros: More gradual feature transformation
   - Cons: Increased model complexity

#### Training Dynamics

