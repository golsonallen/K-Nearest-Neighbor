K-Nearest Neighbors Program
===========================


By Griffin Olson-Allen <griffino@umich.edu>

This program implements the k-nearest neighbors algorithm from scratch, allowing the user to
specify the K value. It includes the Iris flower data set and red wine quality dataset but can be 
applied to any dataset. The program takes 3 command-line arguments: the KNN executable, 
the training file, and the testing file.

I split both original sets into training and testing data, 70%, for training, and 30% for testing. 
For the Iris flower testing data I randomly selected instances from the training data and then 
shuffled them. For the wine quality testing data I used the last 480 entries. 

Thank you to Martin Broadhurst for providing the split function used to parse the CSV file 
headers into individual attributes. 
Link: <http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html>

