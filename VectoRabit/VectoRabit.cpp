#include <iostream>
#include <Eigen/Dense>
#include "crow.h"
#include "BallTree.h"

using namespace std;
using namespace Eigen;

int main() {
    // Generate random data
    MatrixXd data = MatrixXd::Random(100, 3);

    // Build BallTree
    BallTree tree(data);

    crow::SimpleApp app;

    CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)
        ([&tree](const crow::request& req) {
        // Parse the JSON request
        auto json = crow::json::load(req.body);

        // Parse and populate the query point from the JSON data
        VectorXd query_point(3);
        try {
            for (int i = 0; i < 3; ++i) {
                query_point(i) = json["query_point"][i].d();
            }
        }
        catch (exception& e) {
            return crow::response(400, "Error parsing query_point data");
        }

        // Find the k most similar vectors
        int k = 5;
        vector<VectorXd> nearest_neighbors = tree.kNearestNeighbors(query_point, k);
        // Print the results
        cout << "Query point:\n" << query_point << "\n\n";
        cout << k << " nearest neighbors:\n";
        for (int i = 0; i < nearest_neighbors.size(); ++i) {
            cout << "Neighbor " << i + 1 << ":\n" << nearest_neighbors[i].transpose() << "\n";
        }

        // Prepare the JSON response
        crow::json::wvalue response_json;
        for (int n = 0; n < k; n++) {
            vector<double> vec;
            for (int i = 0; i < nearest_neighbors[i].size(); i++) {
                vec.push_back(nearest_neighbors[n][i]);
            }
            response_json["neighbors"][n] = vec;
        }




        return crow::response(response_json);
            });

    app.port(8080).run();

    return 0;
}