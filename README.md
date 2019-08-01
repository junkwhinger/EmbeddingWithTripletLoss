# EmbeddingWithTripletLoss
Image Embedding on 2D space with Triplet Loss



## Idea

- TripletLoss
  - learns to place objects with the same class together and separate the ones with different classes
  - widely used in face recognition tasks
- Transfer Learning + TripletLoss = learn image embedding with pre-trained weights and small dataset



## How…?

- Various types of Triplet Mining strategies… which one is gonna be the best?
- Which image does the model find particularly hard to embed?
- How do the embedding information change over training steps?



## Structure

- triplet_miner.py: various mining strategy
- train.py: train network that embeds images to 2d space
- Inference.py: generate scatter plots of embedded images & gifs



## Reference

- triplet.. stuff
- github… online triplet miner code..