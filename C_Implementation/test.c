#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
float max_value=0;
float min_value=0;

void findextreme(float value) {
    if (value > max_value) {
        max_value = value;
    }
    if (value < min_value) {
        min_value = value;
    }
}

float* npConv_one_filter1D(float* img, int length, float* weight_one_filter, int kernel_size) {
    float* result = (float*)malloc(length * sizeof(float));
    for (int i = 0; i < length; i++) {
        float output_value = 0;
        for (int m = 0; m < kernel_size; m++) {
            int weight_index = m;
            int input_index = i + m;
            output_value += weight_one_filter[weight_index] * img[input_index];
            findextreme(output_value);
        }
        result[i] = output_value;
        findextreme(result[i]);
    }
    return result;
}

float* npConv1D(float* feature, int length, float* weight, float* bias, int in_channels, int out_channels, int kernel_size, int padding, int group) {
    float* result = (float*)malloc(out_channels * length * sizeof(float));
    for(int i=0;i<out_channels * length;i++){
        result[i] = 0.0;
    }
    if (padding != 0) {
        float* pad = (float*)malloc((length + 2 * padding) * in_channels * sizeof(float));
        for (int i = 0; i < in_channels; i++) {
            for (int j = 0; j < padding; j++) {
                pad[i * (length + 2 * padding) + j] = 0;
                pad[i * (length + 2 * padding) + padding + length + j] = 0;
            }
            for (int j = 0; j < length; j++) {
                pad[i * (length + 2 * padding) + padding + j] = feature[i * length + j];
            }
        }
        feature = pad;
    }
    if (group == -1) {
        for (int i = 0; i < out_channels; i++) {
            for (int j = 0; j < in_channels; j++) {
                float* weight_one_filter = (float*)malloc(kernel_size * sizeof(float));
                float* newfeature = (float*)malloc((length + 2*padding) * sizeof(float));
                memcpy(weight_one_filter, weight + i * in_channels * kernel_size + j * kernel_size, kernel_size * sizeof(float));
                memcpy(newfeature, feature + j * (length + 2 * padding), (length + 2 * padding) * sizeof(float));
                float* conv_result = npConv_one_filter1D(newfeature, length, weight_one_filter, kernel_size);
                for (int k = 0; k < length; k++) {
                    result[i * length + k] += conv_result[k];
                }
                free(weight_one_filter);
                free(newfeature);
                free(conv_result);
            }
            for (int k = 0; k < length; k++) {
                result[i * length + k] += bias[i];
            }
        }
    } else {
        for (int i = 0; i < in_channels; i++) {
            float* weight_one_filter = (float*)malloc(kernel_size * sizeof(float));
            float* newfeature = (float*)malloc((length + 2*padding) * sizeof(float));
            memcpy(weight_one_filter, weight + i * kernel_size, kernel_size * sizeof(float));
            memcpy(newfeature, feature + i * (length + 2 * padding), (length + 2 * padding) * sizeof(float));
            float* conv_result = npConv_one_filter1D(newfeature, length, weight_one_filter, kernel_size);
            for (int j = 0; j < length; j++) {
                result[i * length + j] = conv_result[j] + bias[i];
            }
            free(weight_one_filter);
            free(newfeature);
            free(conv_result);
        }
    }

    return result;
}

float* npMaxPool1D(float* img, int length, int kernel_size, int stride, int padding, int inchannel) {
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    float* result = (float*)malloc(out_length * inchannel * sizeof(float));

    for (int i = 0; i < inchannel; i++) {
        for (int j = 0; j < out_length; j++) {
            // Calculate the pool window position in the input sequence
            int start = j * stride - padding;
            int end = start + kernel_size;
            // Initialize max value with the minimum possible value
            float max = img[(start < 0) ? 0 : start];
            for (int k = (start < 0) ? 0 : start; k < ((end > length) ? length : end); k++) {
                float current_val = img[k];
                if (current_val > max) {
                    max = current_val;
                }
            }
            result[out_length * i + j] = max;
        }
    }

    return result;
}

float* npAvgPool1D(float* img, int length, int kernel_size, int stride, int padding, int inchannel) {
    int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    float* result = (float*)malloc(out_length * inchannel * sizeof(float));

    for (int i = 0; i < inchannel; i++) {
        float* pad = (float*)malloc((length + 2 * padding) * sizeof(float));
        for (int k = 0; k < padding; k++) {
            pad[k] = 0.0;
            pad[length + padding + k] = 0.0;
        }
        for (int k = padding; k < (padding + length); k++) {
            pad[k] = img[i * length + k - padding];
        }
        for (int j = 0; j < out_length; j++) {
            float sum = 0.0;
            for (int m = 0; m < kernel_size; m++) {
                sum += pad[j * stride + m];
            }
            result[out_length * i + j] = sum / kernel_size;
        }

        free(pad);
    }

    return result;
}

float* Linear(float *input_data, int length, int inchannel, int outnumber, float *weights, float *bias) {
    float* output = (float*)malloc(outnumber * sizeof(float));

    for (int k = 0; k < outnumber; k++) {
        output[k] = 0.0;

        for (int i = 0; i < inchannel; i++) {
            for (int j = 0; j < length; j++) {
                output[k] += input_data[i * length + j] * weights[k * length * inchannel + i * length + j];
            }
        }

        output[k] += bias[k];
    }
    return output;
}

void readCSV(const char* filename, float* array, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return;
    }

    char line[MAX_LINE_LENGTH];
    int i = 0;

    while (fgets(line, MAX_LINE_LENGTH, file) && i < n) {
        char* token = strtok(line, ",");
        while (token != NULL && i < n) {
            array[i] = atof(token);
            i++;
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
}

float findMaxIndex(float* arr) {
    float max = arr[0];
    int index = 0;

    for (int i = 0; i < 5; i++) {
        if (arr[i] > max) {
            max = arr[i];
            index = i;
        }
    }
    return index;
}

void divideBy256(float arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] /= 256.0;
    }
}

float* np_nn(float* signal) {
    float* conv1_weight = (float*)malloc(15 * sizeof(float));
    float* conv1_bias = (float*)malloc(1 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv1_weight.txt", conv1_weight, 15);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv1_bias.txt", conv1_bias, 1);
    divideBy256(conv1_weight,15);
    divideBy256(conv1_bias,1);
    float* conv1 = npConv1D(signal, 300, conv1_weight, conv1_bias, 1, 1, 15, 7, -1);
    conv1 = npMaxPool1D(conv1, 300, 3, 2, 1, 1);
    free(conv1_weight);
    free(conv1_bias);

    float* conv2_weight = (float*)malloc(17 * sizeof(float));
    float* conv2_bias = (float*)malloc(1 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv2_weight.txt", conv2_weight, 17);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv2_bias.txt", conv2_bias, 1);
    divideBy256(conv2_weight,17);
    divideBy256(conv2_bias,1);
    float* conv2 = npConv1D(conv1, 150, conv2_weight, conv2_bias, 1, 1, 17, 8, -1);
    conv2 = npMaxPool1D(conv2, 150, 3, 2, 1, 1);
    free(conv2_weight);
    free(conv2_bias);
    free(conv1);

    float* conv3_weight = (float*)malloc(38 * sizeof(float));
    float* conv3_bias = (float*)malloc(2 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv3_weight.txt", conv3_weight, 38);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv3_bias.txt", conv3_bias, 2);
    divideBy256(conv3_weight,38);
    divideBy256(conv3_bias,2);
    float* conv3 = npConv1D(conv2, 75, conv3_weight, conv3_bias, 1, 2, 19, 9, -1);
    conv3 = npAvgPool1D(conv3, 75, 3, 2, 1, 2);
    free(conv3_weight);
    free(conv3_bias);
    free(conv2);

    float* conv4_weight = (float*)malloc(42 * sizeof(float));
    float* conv4_bias = (float*)malloc(2 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv4_weight.txt", conv4_weight, 42);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv4_bias.txt", conv4_bias, 2);
    divideBy256(conv4_weight,42);
    divideBy256(conv4_bias,2);
    float* conv4 = npConv1D(conv3, 38, conv4_weight, conv4_bias, 2, 2, 21, 10, 0);
    free(conv4_weight);
    free(conv4_bias);
    free(conv3);

    float* conv5_weight = (float*)malloc(8 * sizeof(float));
    float* conv5_bias = (float*)malloc(4 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv5_weight.txt", conv5_weight, 8);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/conv5_bias.txt", conv5_bias, 4);
    divideBy256(conv5_weight,8);
    divideBy256(conv5_bias,4);
    float* conv5 = npConv1D(conv4, 38, conv5_weight, conv5_bias, 2, 4, 1, 0, -1);
    free(conv5_weight);
    free(conv5_bias);
    free(conv4);

    float* fc1_weight = (float*)malloc(2432 * sizeof(float));
    float* fc1_bias = (float*)malloc(16 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/fc1_weight.txt", fc1_weight, 2432);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/fc1_bias.txt", fc1_bias, 16);
    divideBy256(fc1_weight,2432);
    divideBy256(fc1_bias,16);
    float* fc1 = Linear(conv5, 38, 4, 16, fc1_weight, fc1_bias);
    free(fc1_weight);
    free(fc1_bias);
    free(conv5);

    for (int i = 0; i < 16; i++) {
        if (fc1[i] < 0) {
            fc1[i] = 0;
        }
    }

    float* fc2_weight = (float*)malloc(80 * sizeof(float));
    float* fc2_bias = (float*)malloc(5 * sizeof(float));
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/fc2_weight.txt", fc2_weight, 80);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/fc2_bias.txt", fc2_bias, 5);
    divideBy256(fc2_weight,80);
    divideBy256(fc2_bias,5);
    float* fc2 = Linear(fc1, 16, 1, 5, fc2_weight, fc2_bias);
    free(fc2_weight);
    free(fc2_bias);
    free(fc1);

    return fc2;
}

int main() {
    int correct_num = 0;
    int totalnum = 1000;

    float* ECGsignal = (float*)malloc(300 * totalnum * sizeof(float));
    float* labels = (float*)malloc(totalnum * sizeof(float));
    float* signal = (float*)malloc(300 * sizeof(float));

    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/X_test_np.txt", ECGsignal, 300 * totalnum);
    readCSV("C:/Users/HeZhuoLi/Desktop/ECG_recognition_c/Y_test_np.txt", labels, totalnum);
    divideBy256(ECGsignal,300 * totalnum);
    for(int i=0;i<totalnum;i++){
        memcpy(signal, ECGsignal + i*300, 300 * sizeof(float));
        float* my_list = np_nn(signal);
        int Index = findMaxIndex(my_list);
        if(Index == (int)labels[i]){
            correct_num ++;
        }else{
            printf("err_index:%d ",i);
            printf("predict:%d ", Index);
            printf("lable:%d\n",(int)labels[i]);
        }
    }
    float rate = ((float)correct_num / (float)totalnum) * 100;
    printf("accuary:%.2f", rate);

    free(ECGsignal);
    free(labels);
    free(signal);
    return 0;
}