#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

class my_RNN : public NeuralNetwork<float>{
private:
public:
    my_RNN(vector<Tensorholder<float> *> x, vector<Tensorholder<float> *> label) {

        int ptime = 28;
        int batch_size = 28;

        for(int i = 0; i < ptime; i++){
            SetInput(2, x[i], label[i]);
        }

        vector<Operator<float> *> out;

        for(int i = 0; i < batch_size; i++){

          out = new ReShape<float>(x[i], 28, 28, "Flat2Image");   // MNIST 기준

          // ======================= Embedding Layer ======================= //


          // ======================= RNN Layer ========================= //
          out = new RecurrentLayer<float>(out, ptime, 10, 3, 3, 1, 1, 0, FALSE, "Conv_1");
          out = new Tanh<float>(hidden, "Tanh");

          // ======================= Affine Layer ======================= f(Wx+b) 형태로 output을 내는 것.
          out = new ReShape<float>(out, 1, 1, 5 * 5 * 20, "Image2Flat");

          // ======================= Softmax Layer =======================
          out = new Linear<float>(out, 5 * 5 * 20, 1024, TRUE, "Fully-Connected_1");

          out = new Relu<float>(out, "Relu_3");
          //
          //// ======================= FC Layer =======================
          out = new Linear<float>(out, 1024, 10, TRUE, "Fully-connected_2");

          AnalyzeGraph(out);

          // ======================= Select LossFunction Function ===================
          // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
          // SetLossFunction(new MSE<float>(out, label, "MSE"));
          SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
          // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

          // ======================= Select Optimizer ===================
          SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
          // SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
          // SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
          // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
          // SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));

















        }

    }

    virtual ~my_CNN() {}
};
