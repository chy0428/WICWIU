#include "ImageNetReader.h"

#include <unistd.h>

int main(int argc, char const *argv[]) {
    std::cout << "======================START=========================" << '\n';

    ImageNetDataReader<float> *data_reader = new ImageNetDataReader<float>(200, 10, TRUE);

    Tensor<float> **data = NULL;

    sleep(3);

    data = data_reader->GetDataFromBuffer();

    std::cout << data[1]->GetShape() << '\n';
    std::cout << data[0]->GetShape() << '\n';

    delete data[0];
    delete data[1];

    delete data;

    sleep(3);

    data = data_reader->GetDataFromBuffer();

    std::cout << data[1]->GetShape() << '\n';
    std::cout << data[0]->GetShape() << '\n';

    delete data[0];
    delete data[1];

    delete data;

    sleep(3);

    data_reader->StopDataPreprocess();

    sleep(3);

    delete data_reader;
    
    std::cout << "======================Done=========================" << '\n';

    return 0;
}