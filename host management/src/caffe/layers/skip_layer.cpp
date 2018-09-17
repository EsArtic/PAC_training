#include <cstdio>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SkipLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    data_type_ = this->layer_param_.skip_param().skip_type();

    int T_ = bottom[0]->shape(0);
    int N_ = bottom[0]->shape(1);

    if ((data_type_ == 0) || (data_type_ == 1)) {
        CHECK_GE(bottom[0]->num_axes(), 2)
            << "bottom[0] must have at least 2 axes -- (#timesteps, #streams, ...)";
        LOG(INFO) << "Initializing skip layer: assuming input batch contains "
            << T_ << " timesteps of " << N_ << " independent streams.";
    } else {
        CHECK_EQ(bottom[0]->num_axes(), 2)
            << "bottom[1] must have exactly 2 axes -- (#timesteps, #streams)";
    }

    skip_length_ = this->layer_param_.skip_param().skip_length();
    Reshape(bottom, top);
}

template <typename Dtype>
void SkipLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    if (data_type_ == 0) {
        vector<int> top_shape(3);
        top_shape[0] = bottom[0]->shape(0) / skip_length_;
        top_shape[1] = bottom[0]->shape(1);
        top_shape[2] = bottom[0]->shape(2);
        top[0]->Reshape(top_shape);
    } else if (data_type_ == 1) {
        vector<int> top_shape(2);
        top_shape[0] = bottom[0]->shape(1);
        top_shape[1] = bottom[0]->shape(2);
        top[0]->Reshape(top_shape);
    } else {
        vector<int> top_shape(2);
        top_shape[0] = bottom[0]->shape(0) / skip_length_;
        top_shape[1] = bottom[0]->shape(1);
        top[0]->Reshape(top_shape);
    }
}

template <typename Dtype>
void SkipLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int delta;
    if ((data_type_ == 0) || (data_type_ == 1)) {
        delta = bottom[0]->shape(1)*bottom[0]->shape(2);
    } else {
        delta = bottom[0]->shape(1);
    }

    if ((data_type_ == 0) || (data_type_ == 2)) {
        for (int i = 0; i < bottom[0]->shape(0); i++) {
            if ((i != 0) && (i % skip_length_ == 0)) {
                for (int j = 0; j < delta; j++) {
                    *top_data = *(bottom_data + (i * delta + j));
                    if (j == 0 && data_type_ == 2) *top_data = 0;
                    top_data++;
                }
            }
        }
    } else {
        int offset = bottom[0]->shape(0) - 1;
        for (int i = 0; i < delta; i++) {
            *top_data = *(bottom_data + (offset * delta + i));
            top_data++;
        }
    }
}

template <typename Dtype>
void SkipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        int delta;
        if ((data_type_ == 0) || (data_type_ == 1)) {
            delta = bottom[0]->shape(1)*bottom[0]->shape(2);
        } else {
            delta = bottom[0]->shape(1);
        }

        if ((data_type_ == 0) || (data_type_ == 2)) {
            int index = 0;
            for (int i = 0; i < bottom[0]->shape(0); i++) {
                if ((i != 0) && (i % skip_length_ == 0)) {
                    for (int j = 0; j < delta; j++) {
                        bottom_diff[i * delta + j] = top_diff[index];
                        index++;
                    }
                }
            }
        } else {
            for (int i = 0; i < delta; i++) {
                bottom_diff[(bottom[0]->shape(0) - 1) * delta + i] = top_diff[i];
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(SkipLayer);
#endif

INSTANTIATE_CLASS(SkipLayer);
// REGISTER_LAYER_CLASS(SKIP);

}  // namespace caffe
