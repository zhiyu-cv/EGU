# Enhancing Gaussian Utilization for Efficient Real-Time 3D Gaussian Splatting

# Abstract

In recent years, 3D Gaussian Splatting (3DGS) has garnered significant attention for its superior rendering quality and real-time performance. However, the inefficient utilization of Gaussians in 3DGS necessitates the use of millions of primitives to adapt to the geometry and appearance of 3D scenes, leading to significant redundancy. To address this issue, we propose an efficient adaptive density control strategy that incorporates Cross-Section-Oriented splitting and Heterogeneous cloning operations. These modifications prevent the proliferation of redundant Gaussians and improve Gaussian utilization. Furthermore, we introduce opacity adaptive pruning, adaptive thresholds, and Gaussian importance weights to refine the Gaussian selection process. Our post-processing Gaussian refinement pruning further eliminates small-scale and low-opacity Gaussians. Experimental results on various challenging datasets demonstrate that our method achieves state-of-the-art rendering quality while consuming less storage space, reducing the number of Gaussians by up to 42\% compared to 3DGS.

# Runing

We build our code on top of the open-source 3DGS codebase. You can read file ‘3DGS-README.md’ to install and run our code.

