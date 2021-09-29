# K-means-Clustering
**K-means clustering to quantize the MFCC features**


**Problem Statement:**

Experiment with various segment lengths from each VCV recording centered around the middle of the consonant as illustrated in the figure below. A segment (as shown by the red colored part in the figure below) starts from M-p(M-B) and ends at M+p(E-M). Thus, the segment length is parameterized by p (0<p<1). When p=1, the entire VCV from B to E is used. Experiment with four values of p, namely, 0.25, 0.50, 0.75 and 1.0.

For the classification task, consider a 39-dim MFCC with delta and delta delta coefficients computed with a window size of 20msec with a 10msec shift. Thus, each VCV recording will be represented as a sequence of 39-dim features, i.e., a feature matrix, the length of which varies from one recording to another and also depending on the choice of p.

Perform K-means clustering to quantize the MFCC features from all training VCV recordings using the centers of K clusters. Use K=16.
