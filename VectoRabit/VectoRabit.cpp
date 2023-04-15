#include <iostream>
#include <Eigen/Dense>
#include "crow.h"
#include "BallTree.h"

using namespace std;
using namespace Eigen;

int main()
{
    // Initialize data
    MatrixXd data(0, 0);

    // Build BallTree
    BallTree tree(data);

    // Set vector size
    int VECTOR_SIZE = 3;

    crow::SimpleApp app;

    CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)([&tree, VECTOR_SIZE](const crow::request &req)
                                                               {
        // Parse the JSON request
        auto json = crow::json::load(req.body);

        // Parse and populate the query point from the JSON data
        VectorXd query_point(VECTOR_SIZE);
        try {
            for (int i = 0; i < VECTOR_SIZE; ++i) {
                query_point(i) = json["query_point"][i].d();
            }
        }
        catch (exception& e) {
            return crow::response(400, "Error parsing query_point data");
        }

        int size = tree.size();
        cout << "\n Size: " << size << endl;
        // Find the k most similar vectors
        int k = size < 5 ? size : 5;
        vector<VectorXd> nearest_neighbors = tree.kNearestNeighbors(query_point, k);
        // Print the results
        cout << "Query point:\n" << query_point << "\n\n";
        cout << k << " nearest neighbors:\n";
        for (int i = 0; i < k; ++i) {
            cout << "Neighbor " << i + 1 << ":\n" << nearest_neighbors[i].transpose() << "\n";
        }

        // Prepare the JSON response
        crow::json::wvalue response_json;
        for (int n = 0; n < k; n++) {
            vector<double> vec;
            for (int i = 0; i < nearest_neighbors[n].size(); i++) {
               vec.push_back(nearest_neighbors[n][i]);
            }
            response_json["neighbors"][n] = vec;
        }


        return crow::response(response_json); });

    CROW_ROUTE(app, "/insert").methods(crow::HTTPMethod::POST)([&tree, VECTOR_SIZE](const crow::request &req)
                                                               {
        // Parse the JSON request
        auto json = crow::json::load(req.body);

        // Parse and populate the query point from the JSON data
        VectorXd vector(VECTOR_SIZE);
        try {
            for (int i = 0; i < VECTOR_SIZE; ++i) {
                vector(i) = json["vector"][i].d();
            }
        }
        catch (exception& e) {
            return crow::response(400, "Error parsing vector data");
        }


        double tolerance = 1e-8;
        bool exists = tree.contains(vector, tolerance);
        if (exists) {
            cout << "Exists ";
            crow::response res(crow::status::OK);
            res.body = "Vector is already stored in the database";
            return res;
        }
        cout << "Does not Exist ";
        
        // insert vector
        tree.insert(vector);


        crow::response res(crow::status::CREATED);
        res.body = "Success!";
        return res; });

    app.port(8080).multithreaded().run();

    return 0;
}