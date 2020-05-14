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
    void store_training_data (csvstream &input, vector<string> attribute_names,
                              string label) {
        map<string, string> map;
        while (input >> map) {
            int index = 0;
            for (string &attribute : attribute_names) {
                attributes[index].push_back(stod(map[attribute]));
                index++;
            }
            classes.push_back(map[label]);
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
            // between each flower in the training set and the test flower
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
    
    // Prompt user for K and other info about dataset
    int K = 0;
    cout << "Please enter an odd number for K: ";
    cin >> K;
    
    int num_instances = 0;
    cout << "Please enter the number of instances in the TRAINING dataset"
        << " (size): ";
    cin >> num_instances;
    
    int num_attributes = 0;
    cout << "Please enter the number of attributes in the dataset"
        << " (input variables): ";
    cin >> num_attributes;
    
    
    vector<string> attributes;
    cout << "Please enter the attributes in the order they appear in the train"
        << " and test file headers." << endl;
    for (int i = 1; i <= num_attributes; ++i) {
        string attribute;
        cout << "Attribute #" << i << ": ";
        cin >> attribute;
        attributes.push_back(attribute);
    }
    
    string label;
    cout << "Please enter the label or tag (output variable): ";
    cin >> label;
    
    KNN classifier(K, num_attributes, num_instances);
    classifier.store_training_data(train, attributes, label);
    
    // process testing data
    map<string, string> row;
    cout << endl;
    cout << "RESULTS" << endl;
    while (test >> row) {
    
        vector<double> test_flower;
        for (string attribute : attributes) {
            test_flower.push_back(stod(row[attribute]));
        }
        
        string actual_label = row["class"];
        classifier.store_distance(test_flower);
        classifier.run_classifier(actual_label);
    }
    classifier.print_accuracy();
}
