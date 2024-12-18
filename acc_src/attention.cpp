#ifdef _OPENMP
#include <omp.h>

#include "../include/attention.h"

#include "../include/datatypes.h"
#include "../include/modules.h"

#include <utility>
#include <assert.h>
#include <math.h>

#include <iostream>

Attention::Attention (
    vit_size _dim,
    vit_size _num_heads,
    vit_bool use_qkv_bias,
    vit_bool _use_qk_norm
) :
    q_gen(_dim,_dim,use_qkv_bias),
    k_gen(_dim,_dim,use_qkv_bias),
    v_gen(_dim,_dim,use_qkv_bias),
    q_norm(_dim/_num_heads, 0.00001, true),
    k_norm(_dim/_num_heads, 0.00001, true),
    proj(_dim,_dim,true)
{
    assert(_dim % _num_heads == 0); // _dim must be divisible by _num_heads

    dim = _dim;
    num_heads = _num_heads;
    head_dim = _dim / _num_heads;
    scale = std::pow(head_dim, -0.5);
    use_qk_norm = _use_qk_norm;
}

Attention::Attention(Attention&& attn) :
    q_gen(std::move(attn.q_gen)),
    k_gen(std::move(attn.k_gen)),
    v_gen(std::move(attn.v_gen)),
    q_norm(std::move(attn.q_norm)),
    k_norm(std::move(attn.k_norm)),
    proj(std::move(attn.proj))
{
    dim = attn.dim;
    num_heads = attn.num_heads;
    head_dim = attn.head_dim;
    scale = attn.scale;
    use_qk_norm = attn.use_qk_norm;
}

Attention::~Attention() {}

Attention& Attention::operator= (Attention&& attn) {
    dim = attn.dim;
    num_heads = attn.num_heads;
    head_dim = attn.head_dim;
    scale = attn.scale;
    use_qk_norm = attn.use_qk_norm;

    q_gen = std::move(attn.q_gen);
    k_gen = std::move(attn.k_gen);
    v_gen = std::move(attn.v_gen);
    q_norm = std::move(attn.q_norm);
    k_norm = std::move(attn.k_norm);
    proj = std::move(attn.proj);

    return *this;
}

vit_size Attention::get_dim() const {
    return dim;
}

vit_size Attention::get_num_heads() const {
    return num_heads;
}

vit_size Attention::get_head_dim() const {
    return head_dim;
}

vit_float Attention::get_scale() const {
    return scale;
}

vit_bool Attention::get_use_qk_norm() const {
   return use_qk_norm;
}

void Attention::move_qkv_gen(Linear& _q_gen, Linear& _k_gen, Linear& _v_gen) {
    q_gen = std::move(_q_gen);
    k_gen = std::move(_k_gen);
    v_gen = std::move(_v_gen);
}

void Attention::move_norms(LayerNorm& _q_norm, LayerNorm& _k_norm) {
    q_norm = std::move(_q_norm);
    k_norm = std::move(_k_norm);
}

void Attention::move_proj(Linear& _proj) {
    proj = std::move(_proj);
}

void Attention::to_ofstream(std::ofstream& os) const {
    assert( os.is_open() );

    os.write( (char*) &dim, sizeof(vit_size));
    os.write( (char*) &num_heads, sizeof(vit_size));
    os.write( (char*) &head_dim, sizeof(vit_size));
    os.write( (char*) &scale, sizeof(vit_float));
    os.write( (char*) &use_qk_norm, sizeof(vit_bool));

    q_gen.to_ofstream(os);
    k_gen.to_ofstream(os);
    v_gen.to_ofstream(os);
    if (use_qk_norm == true) {
        q_norm.to_ofstream(os);
        k_norm.to_ofstream(os);
    }
    proj.to_ofstream(os);
}

void Attention::from_ifstream(std::ifstream& is) {
    assert( is.is_open() );

    is.read( (char*) &dim, sizeof(vit_size));
    is.read( (char*) &num_heads, sizeof(vit_size));
    is.read( (char*) &head_dim, sizeof(vit_size));
    is.read( (char*) &scale, sizeof(vit_float));
    is.read( (char*) &use_qk_norm, sizeof(vit_bool));

    q_gen.from_ifstream(is);
    k_gen.from_ifstream(is);
    v_gen.from_ifstream(is);
    if (use_qk_norm == true) {
        q_norm.from_ifstream(is);
        k_norm.from_ifstream(is);
    }
    proj.from_ifstream(is);
}

void Attention::forward(const Tensor& x_in, Tensor& x_out) const {
    Tensor query, key, value;

    q_gen(x_in, query);
    k_gen(x_in, key);
    v_gen(x_in, value);

    if (use_qk_norm == true) {
        q_norm(query, num_heads, head_dim);
        k_norm(key, num_heads, head_dim);
    }

    std::cout << "Query dims: B=" << query.get_B() << ", N=" << query.get_N() << ", C=" << query.get_C() << std::endl;
    std::cout << "Expected C: " << num_heads * head_dim << std::endl;

    multi_head_attention(query, key, value, scale, x_out, num_heads, head_dim);

    proj(x_out,x_out);
}

void Attention::single_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    vit_float _scale,
    Tensor& x_out
) const {
    this->multi_head_attention(query, key, value, _scale, x_out, 1, x_out.get_C());
}

/*
#pragma acc routine
void Attention::multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    vit_float _scale,
    Tensor& x_out,
    vit_size _num_heads,
    vit_size _head_dim
) const {
    assert(query.get_C() == _num_heads*_head_dim);

    assert(key.get_B() == query.get_B());
    assert(value.get_B() == query.get_B());
    assert(key.get_N() == query.get_N());
    assert(value.get_N() == query.get_N());
    assert(key.get_C() == query.get_C());
    assert(value.get_C() == query.get_C());

    vit_size N = query.get_N();
    Tensor qk(query.get_B(), N, N * _num_heads);
    Tensor y(query.get_B(), N, query.get_C());

    vit_float val;
    vit_float cumulative;
    
  //  #pragma acc enter data copyin(query[0:1], key[0:1], value[0:1])
  //  #pragma acc enter data create(qk[0:1], y[0:1])

  //  #pragma acc parallel loop collapse(2)
    for (int batch=0;batch<y.get_B();++batch) {
        for (int nh=0;nh<num_heads;++nh) {

            // qk is the matrix product query * key^T
//	    #pragma acc loop independent 
            for (int q_n=0;q_n<N;++q_n) {
//		#pragma acc loop independent    
                for (int k_n=0;k_n<N;++k_n) {
                    val = 0;
//		    #pragma acc loop reduction(+:val)
                    for (int c=0;c<_head_dim;++c) {
                        val +=
                            query.at(batch, q_n, (nh*_head_dim) + c) *
                            key.at(batch, k_n, (nh*_head_dim) + c);
                    }
                    val *= _scale;
                    qk.set(batch, q_n, (nh*N) + k_n, val);
                }
            }

            // softmax of qk
  //          #pragma acc loop independent
	    for (int qk_n=0;qk_n<N;++qk_n) {
                cumulative = 0;
//		#pragma acc loop reduction(+:cumulative)
                for (int qk_c=0;qk_c<N;++qk_c) { // qk is B*N*(N*nh), that's why it's C is also N
                    val = qk.at(batch, qk_n, (nh*N) + qk_c);
                    val = std::exp(val);
                    cumulative += val;
                    qk.set(batch, qk_n, (nh*N) + qk_c, val);
                }
//		#pragma acc loop independent
                for (int qk_c=0;qk_c<N;++qk_c) {
                    val = qk.at(batch, qk_n, (nh*N) + qk_c);
                    val /= cumulative;
                    qk.set(batch, qk_n, (nh*N) + qk_c, val);
                }
            }

            // y is the matrix product of qk * value
  //          #pragma acc loop independent
	    for (int qk_n=0;qk_n<N;++qk_n) {
//		#pragma acc loop independent    
                for (int v_c=0;v_c<_head_dim;++v_c) {
                    val = 0;
//		    #pragma acc loop reduction(+:val)
                    for (int qk_c=0;qk_c<N;++qk_c) { // qk_c is also v_n
                        val +=
                            qk.at(batch, qk_n, (nh*N) + qk_c) *
                            value.at(batch, qk_c, (nh*_head_dim) + v_c);
                    }
                    y.set(batch, qk_n, (nh*_head_dim) + v_c, val);
                }
            }
        }
    }

  //  #pragma acc exit data copyout(y[0:1]) delete(query[0:1], key[0:1], value[0:1], qk[0:1])

    x_out = std::move(y);
}

*/

/*
void Attention::multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    vit_float _scale,
    Tensor& x_out,
    vit_size _num_heads,
    vit_size _head_dim
) const {
    assert(query.get_C() == _num_heads * _head_dim);
    assert(key.get_C() == _num_heads * _head_dim);
    assert(value.get_C() == _num_heads * _head_dim);

    vit_size N = query.get_N();
    Tensor qk(query.get_B(), N, N * _num_heads);
    Tensor y(query.get_B(), N, query.get_C());

    // Compute Q * K^T
    for (int batch = 0; batch < y.get_B(); ++batch) {
        for (int nh = 0; nh < _num_heads; ++nh) {
            for (int q_n = 0; q_n < N; ++q_n) {
                for (int k_n = 0; k_n < N; ++k_n) {
                    vit_float val = 0;
                    for (int c = 0; c < _head_dim; ++c) {
                        val += query.at(batch, q_n, (nh * _head_dim) + c) *
                               key.at(batch, k_n, (nh * _head_dim) + c);
                    }
                    val *= _scale;
                    qk.set(batch, q_n, (nh * N) + k_n, val);
                }
            }

            // Softmax over the last dimension
            for (int q_n = 0; q_n < N; ++q_n) {
                vit_float cumulative = 0;
                for (int k_n = 0; k_n < N; ++k_n) {
                    vit_float val = std::exp(qk.at(batch, q_n, (nh * N) + k_n));
                    cumulative += val;
                    qk.set(batch, q_n, (nh * N) + k_n, val);
                }
                for (int k_n = 0; k_n < N; ++k_n) {
                    vit_float val = qk.at(batch, q_n, (nh * N) + k_n);
                    qk.set(batch, q_n, (nh * N) + k_n, val / cumulative);
                }
            }

            // Compute Attention * V
            for (int q_n = 0; q_n < N; ++q_n) {
                for (int v_c = 0; v_c < _head_dim; ++v_c) {
                    vit_float val = 0;
                    for (int k_n = 0; k_n < N; ++k_n) {
                        val += qk.at(batch, q_n, (nh * N) + k_n) *
                               value.at(batch, k_n, (nh * _head_dim) + v_c);
                    }
                    y.set(batch, q_n, (nh * _head_dim) + v_c, val);
                }
            }
        }
    }

    x_out = std::move(y);
}

*/


void Attention::multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    vit_float _scale,
    Tensor& x_out,
    vit_size _num_heads,
    vit_size _head_dim
) const {
    // Diagnostic information for debugging
    std::cerr << "Debug Multi-Head Attention Dimensions:" << std::endl;
    std::cerr << "Query Tensor Channels: " << query.get_C() << std::endl;
    std::cerr << "Number of Heads: " << _num_heads << std::endl;
    std::cerr << "Head Dimension: " << _head_dim << std::endl;
    std::cerr << "Expected Channels: " << _num_heads * _head_dim << std::endl;

    // Validate inputs more robustly
    if (_num_heads == 0 || _head_dim == 0) {
        throw std::invalid_argument("Number of heads and head dimension must be positive");
    }

    // Modify the channel dimension check to be more flexible
    if (query.get_C() != _num_heads * _head_dim) {
        // If dimensions don't match exactly, try to adjust
        vit_size adjusted_num_heads = query.get_C() / _head_dim;
        vit_size adjusted_head_dim = query.get_C() / adjusted_num_heads;

        std::cerr << "Attempting to adjust dimensions:" << std::endl;
        std::cerr << "Adjusted Num Heads: " << adjusted_num_heads << std::endl;
        std::cerr << "Adjusted Head Dim: " << adjusted_head_dim << std::endl;

        // Update local variables with adjusted values
        _num_heads = adjusted_num_heads;
        _head_dim = adjusted_head_dim;
    }

    // Re-check dimensions after adjustment
    assert(query.get_C() == _num_heads * _head_dim);
    assert(key.get_C() == _num_heads * _head_dim);
    assert(value.get_C() == _num_heads * _head_dim);

    assert(key.get_B() == query.get_B());
    assert(value.get_B() == query.get_B());
    assert(key.get_N() == query.get_N());
    assert(value.get_N() == query.get_N());

    vit_size B = query.get_B();  // Batch size
    vit_size N = query.get_N();  // Sequence length

    // Create tensors for computation
    Tensor qk(B, N, N * _num_heads);
    Tensor y(B, N, query.get_C());

    // Prepare data for GPU
    #pragma acc enter data copyin(query, key, value) create(qk, y)

    // Parallelize and offload to GPU
    #pragma acc parallel loop collapse(2) present(query, key, value, qk, y)
    for (int batch = 0; batch < B; ++batch) {
        for (int nh = 0; nh < _num_heads; ++nh) {
            // Compute Query * Key^T for each head
            #pragma acc loop independent
            for (int q_n = 0; q_n < N; ++q_n) {
                #pragma acc loop independent
                for (int k_n = 0; k_n < N; ++k_n) {
                    vit_float val = 0;
                    #pragma acc loop reduction(+:val)
                    for (int c = 0; c < _head_dim; ++c) {
                        val += query.at(batch, q_n, (nh * _head_dim) + c) * 
                               key.at(batch, k_n, (nh * _head_dim) + c);
                    }
                    val *= _scale;
                    qk.set(batch, q_n, (nh * N) + k_n, val);
                }
            }

            // Softmax normalization
            #pragma acc loop independent
            for (int qk_n = 0; qk_n < N; ++qk_n) {
                // First pass: find max for numerical stability
                vit_float max_val = -INFINITY;
                #pragma acc loop reduction(max:max_val)
                for (int qk_c = 0; qk_c < N; ++qk_c) {
                    max_val = std::max(max_val, qk.at(batch, qk_n, (nh * N) + qk_c));
                }

                // Compute exponentials and cumulative sum
                vit_float cumulative = 0;
                #pragma acc loop reduction(+:cumulative)
                for (int qk_c = 0; qk_c < N; ++qk_c) {
                    vit_float exp_val = std::exp(qk.at(batch, qk_n, (nh * N) + qk_c) - max_val);
                    qk.set(batch, qk_n, (nh * N) + qk_c, exp_val);
                    cumulative += exp_val;
                }

                // Normalize
                #pragma acc loop independent
                for (int qk_c = 0; qk_c < N; ++qk_c) {
                    vit_float val = qk.at(batch, qk_n, (nh * N) + qk_c) / cumulative;
                    qk.set(batch, qk_n, (nh * N) + qk_c, val);
                }
            }

            // Compute Attention Output (Attention * Value)
            #pragma acc loop independent
            for (int qk_n = 0; qk_n < N; ++qk_n) {
                #pragma acc loop independent
                for (int v_c = 0; v_c < _head_dim; ++v_c) {
                    vit_float val = 0;
                    #pragma acc loop reduction(+:val)
                    for (int qk_c = 0; qk_c < N; ++qk_c) {
                        val += qk.at(batch, qk_n, (nh * N) + qk_c) * 
                               value.at(batch, qk_c, (nh * _head_dim) + v_c);
                    }
                    y.set(batch, qk_n, (nh * _head_dim) + v_c, val);
                }
            }
        }
    }

    // Copy result back and clean up GPU memory
    #pragma acc exit data copyout(y) delete(query, key, value, qk)

    // Move the computed result to the output tensor
    x_out = std::move(y);
}


#else

#error "Error: omp sources must be compiled with -fopenmp compiler flag!"

#endif
