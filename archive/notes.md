# Visualization

* find a way so that you select one pixel in the original image and see which pixels in the reconstructed image are affected by it -> this is like taking one row/column of the attention matrix
* you can take an off the shelf segmentation model and with that create a map of the original image (like color blocks thanks to majority voting) -> then see how does the model perform on that


# Improvements -> IDEA
* make it so that you're not using cutout but extracting them from 360 -> search for preexisting function
* you can compute the n_patches x n_patches distance matrix between the positional embeddings of both images
    - this means that for the aerial image you'll have a 64 x 64 matrix and for the ground image a 256 x 256
    - then you can binarize this matrix by setting all values to 0 except for some elements which will be 1 according to a criterion
        - top k closest elements
        - distance between the embeddings is less than a threshold like 8 pixels

* let's consider the G2A attention matrix to be a 64 x 256 matrix
    - you should normalize it
    - you should compute the cosine similarity of this attention matrix by computing the matmul with its transpose
    - the cosine similarity matrix will be a 64 x 64 matrix

* then, the loss on the attention map is given by the cosine similarity multiplied by the binarized distance matrix
    - this loss should be added to the reconstruction loss

* contrastive loss 