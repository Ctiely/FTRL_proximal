#include "src/dataset.h"
#include "src/model.h"
#include "src/utils.h"

#include <iostream>

using namespace std;

int main() {
    string test = "0 3:1 6:1 17:1 27:1 35:1 40:1 57:1 63:1 69:1 73:1 74:1 76:1 81:1 103:1";
    dataset xtest(test);
    model ftrl(123, 10, 0, 0, 1);
    ftrl.fit("/Users/clytie/Documents/研究生/C++/algorithm/FTRL/data/Traindata.ftrl");
    ftrl.save_model("/Users/clytie/Documents/研究生/C++/algorithm/FTRL/data/model.ftrl");
    model ftrl2("/Users/clytie/Documents/研究生/C++/algorithm/FTRL/data/model.ftrl");
    cout << ftrl.predict(&xtest) << endl;
    cout << ftrl2.predict(&xtest) << endl;
    return 0;
}