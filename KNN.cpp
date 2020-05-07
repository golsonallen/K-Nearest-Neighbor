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
#include <cmath>
#include <vector>
#include "csvstream.h"

using namespace std;

class KNN {
public:
    
    KNN (int k_in) : k(k_in), setosa_tally(0), versicolor_tally(0),
                        virginica_tally(0), label("") {}
    
    void set_label (string label_in) {
        label = label_in;
    }
    
    string get_label() {
        return label;
    }
    
    void inc_setosa_tally() {
        setosa_tally++;
    }
    
    void inc_versicolor_tally() {
        versicolor_tally++;
    }
    
    void inc_virginica_tally() {
        virginica_tally++;
    }
    
    int get_setosa_tally() {
        return setosa_tally;
    }
    
    int get_versicolor_tally() {
        return versicolor_tally;
    }
    
    int get_virginica_tally() {
        return virginica_tally;
    }
        
    // calculate distance between test and training flower
    // store distance and the training flowers class
    // test stores the sepal L and W + pedal L and W for one test flower
    void store_distance (csvstream &train, vector<double> test) {
        map<string, string> map;
        while (train >> map) {
            double SL = stod(map["sepal_length"]);
            double SW = stod(map["sepal_width"]);
            double PL = stod(map["petal_length"]);
            double PW = stod(map["petal_width"]);
            
            // computes Euclidean distance
            double dist = pow((SL - test[0]), 2) + pow((SW - test[1]), 2) +
                            pow((PL - test[2]), 2) + pow((PW - test[3]), 2);
            distances.push_back( {sqrt(dist), map["class"]} );
        }
    }

    // use a functor for sorting by the first element in the pair (distance)
    class Distance_Comparator {
    public:
        bool operator () (const pair<double, string> &p1,
                          const pair<double, string> &p2) {
            return p1.first < p2.first;
        }
    };
    
    // sorts the distances vector based on distance
    void sort_distances () {
        sort(distances.begin(), distances.end(), Distance_Comparator());
    }
    
    
    // Iterate through K nearest neighbors to get each neighbors class
    void k_nearest_classes () {
        for (int i = 0; i < k; ++i) {
            add_votes(distances[i].second);
        }
    }
    
    // Add votes for each class
    void add_votes(string &class_type) {
        if (class_type == "iris_setosa") {
            inc_setosa_tally();
        }
        else if (class_type == "iris_versicolor") {
            inc_versicolor_tally();
        }
        else if (class_type == "iris_virginica") {
            inc_virginica_tally();
        }
    }
    
    // Pick the label with the most votes
    void select_label () {
        int most_votes = max(get_setosa_tally(), get_versicolor_tally());
        most_votes = max(most_votes, get_virginica_tally());
        
        if (most_votes == get_setosa_tally()) {
            set_label("iris_setosa");
        }
        else if (most_votes == get_versicolor_tally()) {
            set_label("iris_versicolor");
        }
        else if (most_votes == get_virginica_tally()) {
            set_label("iris_virginica");
        }
    }
    
    void run_classifier(string &label) {
        sort_distances();
        k_nearest_classes();
        select_label();
        print_results(label);
    }
    
    void print_results (string &actual) {
        cout << "Predicted: " << get_label() << "     ";
        cout << "Actual: " << actual << endl;
    }
    
private:
    // store the training data to avoid having to create a new cvstream for
    // every test flower
    
    
    // store the distance between each the test flower and every training flower
    // as well as the training flowers class
    vector<pair<double, string>> distances;
    int k;
    int setosa_tally;
    int versicolor_tally;
    int virginica_tally;
    string label;
    
};

int main() {
    
    csvstream train("iris_flowers_train.csv");
    csvstream test("iris_flowers_test.csv");
    
    // PROMPT USER FOR K
    int K = 9;
    
    map<string, string> row;
    while (test >> row) {

        // compute distances for each individual test flower
        KNN classifier(K);
        vector<double> test_flower = { stod(row["sepal_length"]),
             stod(row["sepal_width"]), stod(row["petal_length"]),
                                       stod(row["petal_width"]) };
        
        string actual_label = row["class"];
        classifier.store_distance(train, test_flower);
        classifier.run_classifier(actual_label);
    }
    
}
