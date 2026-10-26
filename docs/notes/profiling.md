- cuda asynchronous CPU/GPU
- on a random batch of data 
- first warmup (why)
- what is the variability between different runs
- do with and without the warmup steps
- just time separate stages initially - with/wo synchronize

- what is the total time spten on forward pass
- what kernels are taking the most time


- what does torch.compile do



pytorch - 
bytecode
(
    dynamo intercepts, traces operations and creates a computational graph
    optimizes then sends to dispatch
) 
dispatch
aten::s (c++ implementation of ops)
primary ops
cuda launch