#include <torch/extension.h>
#include <curand_kernel.h>

#define C 16
#define HIDDEN 64
// must remain 3 for now
#define KERNEL_SIZE 3
#define FIRE_RATE 1.0f

__device__ inline int unravel_index(int channel, int x, int y, int W, int H) {
    //to access an element at (tx, ty) in channel c, use state[c * W * H + ty * W + tx]
    return channel * W * H + y * W + x;
}

__device__ inline int unravel_index_kernel(int i_x, int i_y, int channel) {
    //to access an element at (i_x, i_y) in channel c, use state[channel * KERNEL_SIZE * 1 * KERNEL_SIZE + i_y * KERNEL_SIZE + i_x]
    return channel * KERNEL_SIZE * KERNEL_SIZE + i_y * KERNEL_SIZE + i_x;
}

__device__ inline int unravel_index_linear0(int in, int out) {
    //to access an element at (i_x, i_y) in channel c, use state[2*C * out + in + 0 + 0]
    return 2*C * out + in;
}

__device__ inline int unravel_index_linear1(int in, int out) {
    //to access an element at (i_x, i_y) in channel c, use state[HIDDEN * out + in + 0 + 0]
    return HIDDEN * out + in;
}

__global__ void nca2d_cuda_kernel(
    const bool* random,
    const float* state, 
    const float* conv_weight, 
    const float* conv_bias, 
    const float* fc0_weight, 
    const float* fc0_bias, 
    const float* fc1_weight, 
    float* new_state,
    int W, 
    int H,
    int out_C) {
    
    
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    long id = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y;

    float out_conv[C];
    float hidden[HIDDEN];


    if (tx < W && ty < H) {
        //curandState random_state;
        //curand_init(seed, (unsigned long long) tx * W + ty, (unsigned long long) 0, &random_state);
        //if(curand_uniform(&random_state) <= FIRE_RATE) {
        //    for(int out = 0; out < out_C; out++) {
        //        new_state[unravel_index(out, tx, ty, W, H)] = state[unravel_index(out+(C - out_C), tx, ty, W, H)];
        //    }
        //    return;
        //}

        if(!random[unravel_index(0, tx, ty, W, H)]) {
            for(int out = 0; out < out_C; out++) {
                new_state[unravel_index(out, tx, ty, W, H)] = state[unravel_index(out+(C - out_C), tx, ty, W, H)];
            }
            return;
        }

        for(int current_channel = 0; current_channel < C; current_channel++) {
            out_conv[current_channel] = 0.0f;
            for(int i_y = 0; i_y < KERNEL_SIZE; i_y++) {
                for(int i_x = 0; i_x < KERNEL_SIZE; i_x++) {
                    // replicate padding
                    //int input_x = std::clamp(tx + i_x-1, 0, W-1);
                    //int input_y = std::clamp(ty + i_y-1, 0, H-1);

                    // reflect padding
                    int input_x = tx + i_x-1;
                    int input_y = ty + i_y-1;
                    if(input_x < 0) {
                        input_x = 1;
                    }
                    else if(input_x >= W) {
                        input_x = W - 2;
                    }
                    if(input_y < 0) {
                        input_y = 1;
                    }
                    else if(input_y >= H) {
                        input_y = H - 2;
                    }


                    int input_index = unravel_index(current_channel, input_x, input_y, W, H);
                    int kernel_index = unravel_index_kernel(i_x, i_y, current_channel);
                    out_conv[current_channel] += state[input_index] * conv_weight[kernel_index];
                }
            }
            out_conv[current_channel] += conv_bias[current_channel];
        }

        for (int out = 0; out < HIDDEN; out++) {
            hidden[out] = 0.0f;
            for(int in = 0; in < C; in++) {
                int input_index = unravel_index(in, tx, ty, W, H);
                hidden[out] += state[input_index] * fc0_weight[unravel_index_linear0(in, out)];
            }
            for(int in = 0; in < C; in++) {
                hidden[out] += out_conv[in] * fc0_weight[unravel_index_linear0(in+C, out)];
            }
            hidden[out] += fc0_bias[out];
            
            //apply ReLU
            hidden[out] = hidden[out] > 0 ? hidden[out] : 0;
        }
        //for(int i=0;i<HIDDEN;i++) {
        //    temp[unravel_index(i, tx, ty, W, H)] = hidden[i];
        //}
        //return;

        for (int out = 0; out < out_C; out++) {
            int out_index = unravel_index(out, tx, ty, W, H);
            float res = 0.0f;
            for(int in = 0; in < HIDDEN; in++) {
                res += hidden[in] * fc1_weight[unravel_index_linear1(in, out)];
            }
            new_state[out_index] = res + state[unravel_index(out+(C - out_C), tx, ty, W, H)];
        }
        //new_state[unravel_index(0, tx, ty, W, H)] = hidden[0];
        //new_state[unravel_index(1, tx, ty, W, H)] = hidden[1];
        //return;
        
    }
}


torch::Tensor nca2d_cuda(
    const torch::Tensor& random,
    const torch::Tensor& state, 
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& fc0_weight,
    const torch::Tensor& fc0_bias,
    const torch::Tensor& fc1_weight
    ) {
    TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(fc0_weight.is_cuda(), "fc0_weight must be a CUDA tensor");
    TORCH_CHECK(fc0_bias.is_cuda(), "fc0_bias must be a CUDA tensor");
    TORCH_CHECK(fc1_weight.is_cuda(), "fc1_weight must be a CUDA tensor");

    
    TORCH_CHECK(random.is_contiguous(), "random must be contiguous");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(conv_weight.is_contiguous(), "conv_weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(fc0_weight.is_contiguous(), "fc0_weight must be contiguous");
    TORCH_CHECK(fc0_bias.is_contiguous(), "fc0_bias must be contiguous");
    TORCH_CHECK(fc1_weight.is_contiguous(), "fc1_weight must be contiguous");

    int B = state.size(0);
    //int C = state.size(1);
    int H = state.size(2);
    int W = state.size(3);

    int out_C = fc1_weight.size(0);

    TORCH_CHECK(state.size(1) == C, "State channel size mismatch");

    //int nca_hidden_size = fc0_weight.size(0);
    TORCH_CHECK(fc0_weight.size(0) == HIDDEN, "FC0 weight size mismatch");

    TORCH_CHECK(B == 1, "Only batch size 1 is supported");
    TORCH_CHECK(conv_weight.size(0) == C, "Conv weight size mismatch");
    TORCH_CHECK(conv_weight.size(1) == 1, "Conv weight size mismatch");
    TORCH_CHECK(conv_weight.size(2) == KERNEL_SIZE, "Conv weight size mismatch");
    TORCH_CHECK(conv_weight.size(3) == KERNEL_SIZE, "Conv weight size mismatch");

    TORCH_CHECK(conv_bias.size(0) == C, "Conv bias size mismatch");

    TORCH_CHECK(fc0_weight.size(1) == 2*C, "FC0 weight size mismatch");
    TORCH_CHECK(fc0_weight.size(2) == 1, "FC0 weight size mismatch");
    TORCH_CHECK(fc0_weight.size(3) == 1, "FC0 weight size mismatch");

    TORCH_CHECK(fc0_bias.size(0) == HIDDEN, "FC0 bias size mismatch");

    TORCH_CHECK(fc1_weight.size(1) == HIDDEN, "FC1 weight size mismatch");
    TORCH_CHECK(fc1_weight.size(2) == 1, "FC1 weight size mismatch");
    TORCH_CHECK(fc1_weight.size(3) == 1, "FC1 weight size mismatch");

    auto new_state = torch::zeros({1, out_C, H, W}, state.options());

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x,
                (H + threads.y - 1) / threads.y);

    nca2d_cuda_kernel<<<blocks, threads>>>(
        random.data_ptr<bool>(),
        state.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        fc0_weight.data_ptr<float>(),
        fc0_bias.data_ptr<float>(),
        fc1_weight.data_ptr<float>(),
        new_state.data_ptr<float>(),
        W, 
        H,
        out_C);

    return new_state;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nca2d_cuda", &nca2d_cuda, "2D NCA CUDA");
}
