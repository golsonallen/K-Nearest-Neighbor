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
    
    // stores the training data
    void store_training_data (csvstream &input) {
        map<string, string> map;
        int index = 0;
        while (input >> map) {
            SL[index] = stod(map["sepal_length"]);
            SW[index] = stod(map["sepal_width"]);
            PL[index] = stod(map["petal_length"]);
            PW[index] = stod(map["petal_width"]);
            classes[index] = map["class"];
            ++index;
        }
    }
    
    // computes the euclidean distance between 2 values
    double euclidean_dist (double v1, double v2) {
        return sqrt(pow((v1 - v2), 2));
    }
        
    // calculate distance between test and training flower
    // test stores the sepal L and W + pedal L and W for one test flower
    void store_distance (vector<double> test) {
        for (int i = 0; i < TRAINING_DATA_SIZE; ++i) {
            double dist = euclidean_dist(SL[i], test[0]) +
                          euclidean_dist(SW[i], test[1]) +
                          euclidean_dist(PL[i], test[2]) +
                          euclidean_dist(PW[i], test[3]);
            // store the distance between each training flower and test flower
            // and the class of each training flower
            distances.push_back({dist, classes[i]});
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
        clear_old_test_flower();
    }
    
    void print_results (string &actual) {
        cout << "Predicted: " << get_label() << "     ";
        cout << "Actual: " << actual << endl;
    }
    
    // Reuse the stores training data but need to compute new distances
    // for each test flower
    void clear_old_test_flower() {
        distances.clear();
        setosa_tally = 0;
        versicolor_tally = 0;
        virginica_tally = 0;
    }
    
private:
    // store the training data
    const static int TRAINING_DATA_SIZE = 120;
    double SL[TRAINING_DATA_SIZE];
    double SW[TRAINING_DATA_SIZE];
    double PL[TRAINING_DATA_SIZE];
    double PW[TRAINING_DATA_SIZE];
    // store every flowers class
    string classes[TRAINING_DATA_SIZE];
    
    // store the distance between the test flower and every training flower
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
    int K = 0;
    cout << "Please enter a value for K: ";
    cin >> K;
    KNN classifier(K);
    classifier.store_training_data(train);
    
    // process testing data
    map<string, string> row;
    while (test >> row) {
        // compute distances for each individual test flower
        vector<double> test_flower = { stod(row["sepal_length"]),
             stod(row["sepal_width"]), stod(row["petal_length"]),
                                       stod(row["petal_width"]) };
        
        string actual_label = row["class"];
        classifier.store_distance(test_flower);
        classifier.run_classifier(actual_label);
    }
}
