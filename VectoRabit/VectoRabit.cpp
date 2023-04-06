#include <iostream>
#include <Eigen/Dense>

#include "BallTree.h"

using namespace std;


int main() {
    // Generate random data
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(100, 3);

    // Build BallTree
    BallTree tree(data);

    // Define a query point
    Eigen::VectorXd query_point(3);
    query_point << 0.5, 0.5, 0.5;

    // Find the 5 most similar vectors
    int k = 5;
    std::vector<Eigen::VectorXd> nearest_neighbors = tree.kNearestNeighbors(query_point, k);

    // Print the results
    std::cout << "Query point:\n" << query_point << "\n\n";
    std::cout << k << " nearest neighbors:\n";
    for (int i = 0; i < nearest_neighbors.size(); ++i) {
        std::cout << "Neighbor " << i + 1 << ":\n" << nearest_neighbors[i].transpose() << "\n";
    }

    return 0;
}