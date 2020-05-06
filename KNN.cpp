//
//  KNN.cpp
//  KNN Classifier
//
//  Created by Griffin Olson-Allen on 5/6/20.
//  Copyright © 2020 Griffin Olson-Allen. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <map>
#include "csvstream.h"

using namespace std;

class KNN {
public:
    
    
    
    
private:
    
    
};

int main() {
    csvstream input("iris_flowers.csv");
    map<string, string> map;
    while (input >> map) {
        cout << map["sepal_length"] << endl;
    }
}
