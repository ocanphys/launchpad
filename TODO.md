custom validation batches
compute for the entire validation set in batches. This could go into the evaluate function.
later on we will try to get more granular information.

smaller batches can be combined for a larger batch size, controlling the noise.
if you can tag the batches by topic, language, etc then you get provenance and performance metric data. This also goes with training batches. Every sequence that you pass in could have a label so you're really looking at a time series of tag/provenance info. These can be associated with gradients and you can compare the gradients coming from training batches to the gradients from validation batches and compute how these interact in the parameter space...
