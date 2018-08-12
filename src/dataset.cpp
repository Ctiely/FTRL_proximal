//
// Created by Clytie on 2018/8/12.
//

//there must not be a ' ' before char '\n'.
//use '-' to represent a unknown sample.
//For example "# 2:1 3:2.1 5:0.2 14:1\n" or "-1 2:1 3:2.1 5:0.2 14:1\n" or "0 2:1 3:2.1 5:0.2 14:1 \n"
//it will transform -1 to 0
#include "dataset.h"
#include "utils.h"

using namespace std;

dataset::dataset(string & dataLine)
        : dataLine(dataLine) {
    readLine();
    length = (unsigned int)data.indexs.size();
}

void dataset::readLine() {
    const char * cdata = dataLine.c_str();
    unsigned int begin = 0;
    if (cdata[begin] == '-') {
        label = -1;
    } else {
        label =  atoi(cdata);
        //label = (label < 0) ? 0 : label;
    }
    ++begin;
    while (cdata[begin] == ' ') {
        ++begin;
    }
    data = readFromlibsvm<float>(cdata + begin);
}