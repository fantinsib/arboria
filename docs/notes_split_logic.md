
# Split Code Logic : 

We keep a vector of idx throughout the training that references the index of each row of the samples (`std::vector<int> sample_idx` of size `DataSet.n_rows()`)

This idx vector is initialized with std::iota (`std::iota(sample_idx.begin(), sample_idx.end(), 0);`)


When testing a split, std::partition is used to reorder the current node’s subrange of the sample_idx (depending on if it is lower or greater than the tested threshold).

At each node in the training :

- The threshold vector is generated from DataSet.X_ depending on the algorithm used
- DataSet.X_ and DataSet.Y_ are passed as reference to the splitting function. 
- Iterate through threshold -> for each threshold, std::partition is applied to the current node’s subrange of sample_idx. From there, we can use a std::span to pass a view of the split subrange sample_index to each of the next nodes. 
