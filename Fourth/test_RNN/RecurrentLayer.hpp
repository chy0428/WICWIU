#ifndef __RECURRENT_LAYER__
#define __RECURRENT_LAYER__    value

#include "../Module.hpp"

/*!
@class RecurrentLayer Operator들을 그래프로 구성해 RecurrentLayer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 RecurrentLayer의 기능을 수행한다
*/
template<typename DTYPE> class RecurrentLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief RecurrentLayer 클래스 생성자
    @details RecurrentLayer 클래스의 Alloc 함수를 호출한다.
    */
    RecurrentLayer(vector<Operator<DTYPE> *> pInput, int ptime, int input_size, int hidden_size, int batch_size, std::string pName = "NO NAME") : Module<DTYPE>(pName){
        Alloc(pInput, ptime, input_size, hidden_size, batch_size,pName);
    }

    /*!
    @brief RecurrentLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.
    */
    virtual ~RecurrentLayer() {}

    /*!
    @brief 2D RecurrentLayer 그래프를 동적으로 할당 및 구성하는 메소드
    @param pInput 인풋을 벡터로 가지고 있다. {x0, x1, x2, ... , xn}
    @param ptime time step 수
    @param input_size 입력 벡터의 차원수
    @param hidden_size 은닉 상태 벡터의 차원 수
    @param batch_size 미니배치 크기
    @param pName Module의 이름
    @return TRUE
    */
    int Alloc(vector<Operator<DTYPE> *> pInput, int ptime, int input_size, int hidden_size, int batch_size, std::string pName) {

        for(int i=0; i<pInput.size(); i++){
            this->SetInput(pInput[i]);
        }

        vector<Operator<DTYPE> *> hidden = NULL;
        vector<Operator<DTYPE> *> out1 = NULL;
        vector<Operator<DTYPE> *> out2 = NULL;

        Tensorholder<DTYPE> *pWeight_h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1,  batch_size, ptime, hidden_size, hidden_size);
        Tensorholder<DTYPE> *pWeight_x = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1,  batch_size, ptime, input_size, hidden_size);


        // for initialization
        hidden.push_back(new Tensorholder(tensor<DTYPE>::Zeros(1, batch_size, ptime, batch_size, hidden_size);)));

        for(int i = 0; i < batch_size; i++){
          for(int j = 1; j < ptime; j++){
              out1.push_back(new MatMul<DTYPE>(pWeight_h, hidden[j-1]));
              out2.push_back(new MatMul<DTYPE>(pWeight_x, hidden[j]));
              hidden.push_back(new AddChannelWise<DTYPE>(out1, out2));
              this->AnalyzeGraph(hidden[j]);
          }
        }

        return TRUE;
    }
};


#endif  // __RECURRENT_LAYER__
