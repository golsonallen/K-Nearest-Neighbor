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
    
    KNN (int k_in, int atttributes_in, int instances_in)
    : k(k_in), num_attributes(atttributes_in), num_instances(instances_in),
               num_correct(0), num_seen(0), label("") {
            
        for (int i = 0; i < num_attributes; ++i) {
            vector<double> vec;
            attributes.push_back(vec);
        }
    }
    
    void set_label (string label_in) {
        label = label_in;
    }
    
    string get_label() {
        return label;
    }
    
    int get_K() {
        return k;
    }
    
    int get_num_instances() {
        return num_instances;
    }
    
    // stores the training data
    void store_training_data (csvstream &input) {
        map<string, string> map;
        while (input >> map) {
            // generalize number of attributes and name of each attribute
            attributes[0].push_back(stod(map["sepal_length"]));
            attributes[1].push_back(stod(map["sepal_width"]));
            attributes[2].push_back(stod(map["petal_length"]));
            attributes[3].push_back(stod(map["petal_width"]));
            classes.push_back(map["class"]);
        }
    }
    
    // computes the square of the euclidean distance between 2 values
    double euclidean_dist_squared (double x1, double x2) {
        return pow((x1 - x2), 2);
    }
        
    // calculate distance between test and training flower
    // test stores the sepal L and W + pedal L and W for one test flower
    void store_distance (vector<double> test) {
        // loop through each instance
        for (int i = 0; i < num_instances; ++i) {
            // calculate distance of each attribute to get total distance
            double dist = 0.0;
            for (int j = 0; j < num_attributes; ++j) {
                dist += euclidean_dist_squared(attributes[j][i], test[j]);
            }
            distances.push_back({sqrt(dist), classes[i]});
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
    
    // Iterate through K nearest neighbors to get each neighbors class and
    // add a vote for that class
    void k_nearest_classes () {
        for (int i = 0; i < get_K(); ++i) {
            string label = distances[i].second;
            votes[label] += 1;
        }
    }
    
    // Pick the label with the most votes
    void select_label () {
        string best_label = votes.begin()->first;
        int most_votes = votes.begin()->second;
        
        // loop through each label and its votes to determine the label with
        // the most votes
        for (auto &key_val : votes) {
            if (key_val.second > most_votes) {
                most_votes = key_val.second;
                best_label = key_val.first;
            }
        }
        // set the best label
        set_label(best_label);
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
        if (get_label() == actual) {
            num_correct++;
        }
        num_seen++;
    }
    
    void print_accuracy () {
        cout << endl;
        cout << "Accuracy: " << num_correct << "/" << num_seen << endl;
    }
    
    // Reuse the stored training data but need to compute new distances
    // and votes for each test flower
    void clear_old_test_flower() {
        distances.clear();
        votes.clear();
    }
    
private:
    // vector of vectors of doubles to store the data for each attribute
    vector<vector<double> > attributes;
    // store every flowers class
    vector<string> classes;
    
    // store the distance between the test flower and every training flower
    vector<pair<double, string>> distances;
    // map keeps track of each label and how many votes it has to determine
    // the best label
    map<string, int> votes;
    int k;
    int num_attributes;
    int num_instances;
    int num_correct, num_seen;
    string label;
};

int main(int argc, char *argv[]) {
    
    // error message
    if (argc != 3) {
        cout << "Usage: KNN.exe TRAIN_FILE TEST_FILE" << endl;
        return 1;
    }
    
    // opening training file
    ifstream training(argv[1]);
    if (!training.is_open()) {
        cout << "Error opening file: " << string(argv[1]) << endl;
        return 1;
    }
    csvstream train((string(argv[1])));
    
    // opening testing file
    ifstream testing(argv[2]);
    if (!testing.is_open()) {
        cout << "Error opening file: " << string(argv[2]) << endl;
        return 1;
    }
    csvstream test((string(argv[2])));
    
    // PROMPT USER FOR K
    int K = 0;
    cout << "Please enter an odd number for K: ";
    cin >> K;
    
    // FIX
    // SHOULD BE USER INPUT
    KNN classifier(K, 4, 150);
    classifier.store_training_data(train);
    
    // process testing data
    map<string, string> row;
    cout << endl;
    cout << "RESULTS" << endl;
    while (test >> row) {
        /*
        cout << "row:" << "\n";
        for (auto &col:row) {
          const string &column_name = col.first;
          const string &datum = col.second;
          cout << "  " << column_name << ": " << datum << "\n";
        }
        */
        
        // compute distances for each individual test flower
        vector<double> test_flower = { stod(row["sepal_length"]),
             stod(row["sepal_width"]), stod(row["petal_length"]),
                                       stod(row["petal_width"]) };
        
        string actual_label = row["class"];
        classifier.store_distance(test_flower);
        classifier.run_classifier(actual_label);
    }
    classifier.print_accuracy();
}
