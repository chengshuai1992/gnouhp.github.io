---
layout: post
title:  "Relative Margin Loss for Visual-Semantic Embeddings (Draft)"
date:   2019-02-19 15:01:00 -0800
categories: jekyll update


---
*This work is an overview of my own unpublished research. In these open research posts, I will be sharing my thoughts and new ideas on topics that I believe are meaningful contributions, but haven't yet decided if I intend to write a full research paper for. Please see here (link to issues page) if you would like to discuss or collaborate on this research post!*


<br>
## Overview

The most commonly seen loss functions for Visual-Semantic embeddings, such as the hinge-rank loss and the pairwise ranking loss, all utilize an arbitrarily selected margin value, which disregards potentially valuable information of the semantic similarity across data samples. In this post, I give an example of where this shortcoming leads to suboptimal loss values, and I propose the usage of sentence encoders to replace arbitrarily chosen (and constant) margins with relative margins that are specific to the semantic similarity between contrasting image-text pairs.

<br>
## Visual-Semantic Embeddings

Let's start this post with a quick explanation of visual-semantic embeddings. A visual-semantic embedding is a vector space where images and text that contain similar high-level information are mapped to vectors that are also similar to each other. This measure of similarity is most commonly defined as either the cossine similarity (which gives a similarity score based on the angles of 2 vectors) or dot-product similarity (which takes into account both the angle and magnitude of 2 vectors).

Below is an example of the most generic visual-semantic embedding pipeline, where images and text are encoded in parallel (no cross-feature information):

![Generic Joint Embedding Pipeline](/ImageAssets/exampleJE.png)

<br>
## Arbitrarily Selected Margins Aren't Aligned With Human Intuition

In order to train the joint embedding model, the following loss function is commonly used:

*loss(image, caption) =* max *[0, margin - S(U(image), V(caption)) + S(U(image), V(caption<sub>j</sub> ))]*

where:

*U* = Computer Vision Model (i.e. VGG19, ResNet101, SqueezeNet, etc),<br>
*V* = Natural Language Model (i.e. Universal Sentence Encoder, InferSent, Quick-Thoughts, etc),<br>
*S* = Similarity Function (i.e. Cossine Similarity, Dot-Product Similarity, etc), <br>
*caption<sub>j</sub>* = a caption that isn't paired with the image.

Essentially, if the similarity between an image embedding vector, **U(image)**, and one of its true captions, **V(caption)**, is less than a pre-defined distance of the image and a randomly selected caption **V(caption<sub>j</sub>)** from the batch is less than the margin, then this contributes to the pairwise ranking loss. I noticed that there are cases where the one-size-fits all margin choice strategy fails to propagate valuable loss to the model.

Here's example input data that I'll be using to illustrate some points for the rest of the post:

![Sample Image-Caption Data](/ImageAssets/exampleImgCaps.png)
<br>
#### An example where the arbitrary margin is *too large*
Imagine that our model predicts the following:

*S(V(*"Two dogs on the grass on a sunny day"*), U(Image<sub>pugs</sub>)) = 1.0*, and  
*S(V(*"Two dogs on the grass on a sunny day"*), U(Image<sub>retrievers</sub>)) = .95*

This would result in a loss of:

*L =* max *[0, margin - 1.0 + .95]* ,  and assuming *margin = .2*,

*L =* max *[0, .2 - 1.0 + .95] = .15*.


Our model is penalized for having similar vector embeddings for "Two dogs on the grass on a sunny day" and the image of golden retrievers, even though the caption could reasonably describe both images. We would want a *decreased* margin for these types of related image-caption pairs.

<br>
#### An example where the arbitrary margin is *too small*

 Once again, let's look at hypothetical predictions from the model during training:


 *S(V(*"Two dogs on the grass on a sunny day"*), U(Image<sub>pugs</sub>)) = 1.0*, and  
 *S(V(*"Two dogs on the grass on a sunny day"*), U(Image<sub>bread</sub>)) = .79*

 This would result in a loss of:

 *L =* max *[0, margin - 1.0 + .79]* ,  and assuming *margin = .2*,

 *L =* max *[0, .2 - 1.0 + .79] = 0*.

Our model isn't penalized for having similar vector embeddings for "Two dogs on the grass on a sunny day" and the image of bread, even though the caption and the image are completely unrelated. We would want an *increased* margin for completely unrelated image-caption pairs.

Can we do better than an arbitrarily chosen margin value across all image-caption pairs?


<br>
## Relative Margin Loss

How can we get margins that are specific to each unmatched image and caption pair? I propose the usage of universal sentence embeddings, which are recent advances in natural language processing that encode sentences into vectors of a fixed dimension. There are many approaches, from the simplest being the Bag-of-Words baseline, to unsupervised approaches such as Skip-Thought and Quick-Thought vectors, to supervised approaches like InferSent, and finally, to methods that combine both supervised and unsupervised learning, such as Google's Universal Sentence Encoder, which is what I use for the preliminary experiments.

 The Universal Sentence Encoder is publicly available on tfhub and the sentence embeddings are normalized and tuned for the task of semantic similarity, which is what we are interested in finding out. Here are two heatmaps that show the semantic similarities between the captions of the first image against the other two images:

<p float="left">
	<img src="/ImageAssets/heatMapImg.png" width="500"/>
	<img src="/ImageAssets/heatMapImg2.png" width="500"/>
	<figcaption>We can see that the captions concerning the dog images are much more semantically related than the dog/bread pair.</figcaption>
</p>
<br>

I propose the *relative margin loss*, a loss function that takes into consideration the semantic similarity between descriptions across samples in order to get better margin values. I first test a basic, baseline version of the relative margin loss, where I get the candidate margin values by the following equation:

<center> <i>1 - S(Captions<sub>i</sub> , Captions<sub>j</sub> )</i> </center>

where *i* and *j* denote different samples in the dataset.

Here are the relative margin candidate values, high for unrelated caption and image pairs, and lower margins for those more semantically related:

<img src="/ImageAssets/marginMap1.png" alt="drawing" width="500"/>
<img src="/ImageAssets/marginMap2.png" alt="drawing" width="500"/>

From here, we can get the final relative margin values by selecting the minimum of the candidate values for each row, and we're left with the following:

![](/ImageAssets/RelativeMarginValues.png)

<br>
## Preliminary Results

I've tested out the relative margin loss on the Flickr30k and MS-COCO dataset which has 82,000 training images. Both datasets have 5 captions for each image, and test datasets of size 1000. Preliminary results look like my intuition could be correct, and that using relative margins could significantly improve results.

<p float="left">
	<img src="/ImageAssets/mscocoRes.png" width="350"/>
	<img src="/ImageAssets/flickr30kResImg.png" width="350"/>
	<figcaption><i>Solid lines represent the results of models trained with relative margin values, dotted lines show results of using constant margin=.2. Higher values are better.</i></figcaption>
</p>
