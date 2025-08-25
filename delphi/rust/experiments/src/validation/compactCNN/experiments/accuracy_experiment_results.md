Experiment 1: 
---
Validation was run on the model sequentially for CompactCNN on the port 
127.0.0.1:8001 (local) over the course of several hours. 
The accuracy was better than expected given the implementation of batch 
normalization was not yet correct. 

Given baseline experiments with varied batch sizes, and the implementation 
used for this model, I expected an accuracy of around $0.503184$. However,
the model actually achieved an accuracy of $0.53503$ with $168$ correct predictions
out of $314$ samples. The sample-by-sample results can be viewed in the attached
file "classification_results_exp_1.txt"

The model used to achieve this initial result was one which contained the 
folded convolution parameters - 
$$w_{fold} = \gamma \cdot \frac{W}{\sigma{}}