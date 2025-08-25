I have made several updates so far to the code

- added enum for `LinearLayerInfo<F, C>`
    - tried to clone the parameters, but it is causing me trouble
- added `LinearLayer::evaluate_naive()` implementation for batch normalization layer
- added `LinearLayerInfo::evaluate_naive()` implementation for batch normalization layer
    - this doesn't work. I think we may need to perform some preprocessing for this
    - to work correctly. I'm not entirely sure though
- added `LinearLayerInfo.from()` implementation for batch normalization. This function has the same issues which have come up in the other two mentioend above