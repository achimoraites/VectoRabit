#include "BallTree.h"
#include <iostream>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;
using namespace Eigen;

BallTreeNode::BallTreeNode(const MatrixXd &data) : data(data), left(nullptr), right(nullptr)
{
    center = data.colwise().mean();
    radius = (data.rowwise() - center.transpose()).rowwise().norm().maxCoeff();
}

bool BallTreeNode::isLeaf() const
{
    return left == nullptr && right == nullptr;
}

BallTree::BallTree(const MatrixXd &data)
{
    root = buildBallTree(data);
}

BallTreeNode::Ptr BallTree::buildBallTree(const MatrixXd &data)
{
    if (data.rows() == 0)
    {
        return nullptr;
    }

    BallTreeNode::Ptr node = make_shared<BallTreeNode>(data);

    if (data.rows() > 1)
    {
        VectorXd spread = data.colwise().maxCoeff() - data.colwise().minCoeff();
        int widest_dim = max_element(spread.data(), spread.data() + spread.size()) - spread.data();
        double median = data.col(widest_dim).sum() / data.rows();
        vector<MatrixXd> split_data = splitByDimension(data, widest_dim, median);
        node->left = buildBallTree(split_data[0]);
        node->right = buildBallTree(split_data[1]);
    }

    return node;
}

vector<MatrixXd> BallTree::splitByDimension(const MatrixXd &data, int dim, double val)
{
    vector<MatrixXd> result(2);
    vector<int> lower_indices, upper_indices;

    for (int i = 0; i < data.rows(); i++)
    {
        if (data(i, dim) < val)
        {
            lower_indices.push_back(i);
        }
        else
        {
            upper_indices.push_back(i);
        }
    }

    result[0] = data(indicesToSlice(lower_indices), all);
    result[1] = data(indicesToSlice(upper_indices), all);

    return result;
}

VectorXi BallTree::indicesToSlice(const vector<int> &indices)
{
    VectorXi slice(indices.size());
    for (size_t i = 0; i < indices.size(); i++)
    {
        slice(i) = indices[i];
    }
    return slice;
}

vector<VectorXd> BallTree::kNearestNeighbors(const VectorXd &query_point, int k) const
{
    vector<pair<double, VectorXd>> neighbors;
    searchKNearestNeighbors(query_point, k, root, neighbors);

    vector<VectorXd> result;
    for (const auto &neighbor : neighbors)
    {
        result.push_back(neighbor.second);
    }
    return result;
}

void BallTree::searchKNearestNeighbors(const VectorXd &query_point, int k, BallTreeNode::Ptr node, vector<pair<double, VectorXd>> &neighbors) const
{
    if (!node)
    {
        return;
    }

    double dist_to_center = (query_point - node->center).norm();
    double dist_to_farthest = neighbors.empty() ? numeric_limits<double>::max() : neighbors.front().first;

    if (dist_to_center - node->radius > dist_to_farthest)
    {
        return;
    }

    for (int i = 0; i < node->data.rows(); i++)
    {
        double dist = (query_point - node->data.row(i).transpose()).norm();
        if (neighbors.size() < k || dist < dist_to_farthest)
        {
            neighbors.push_back({dist, node->data.row(i)});
            push_heap(neighbors.begin(), neighbors.end(), [](const pair<double, VectorXd> &a, const pair<double, VectorXd> &b)
                      { return a.first < b.first; });
            if (neighbors.size() > k)
            {
                pop_heap(neighbors.begin(), neighbors.end(), [](const pair<double, VectorXd> &a, const pair<double, VectorXd> &b)
                         { return a.first < b.first; });
                neighbors.pop_back();
            }
            dist_to_farthest = neighbors.front().first;
        }
    }

    searchKNearestNeighbors(query_point, k, node->left, neighbors);
    searchKNearestNeighbors(query_point, k, node->right, neighbors);
}

// INSERT
void BallTree::insert(const Eigen::VectorXd &point)
{

    if (!root)
    {
        MatrixXd new_data(1, point.size());
        new_data.row(0) = point;
        root = make_shared<BallTreeNode>(new_data);
    }
    else
    {
        insert(point, root);
    }
}

void BallTree::insert(const Eigen::VectorXd &point, BallTreeNode::Ptr node)
{

    if (node->isLeaf())
    {
        MatrixXd new_data(node->data.rows() + 1, node->data.cols());
        new_data << node->data, point.transpose();
        node->data = new_data;

        node->center = new_data.colwise().mean();
        node->radius = (new_data.rowwise() - node->center.transpose()).rowwise().norm().maxCoeff();

        if (new_data.rows() > 1)
        {
            VectorXd spread = new_data.colwise().maxCoeff() - new_data.colwise().minCoeff();
            int widest_dim = max_element(spread.data(), spread.data() + spread.size()) - spread.data();
            double median = new_data.col(widest_dim).sum() / new_data.rows();
            vector<MatrixXd> split_data = splitByDimension(new_data, widest_dim, median);
            node->left = buildBallTree(split_data[0]);
            node->right = buildBallTree(split_data[1]);
        }
    }
    else
    {
        double dist_left = (point - node->left->center).norm();
        double dist_right = (point - node->right->center).norm();
        if (dist_left < dist_right)
        {
            insert(point, node->left);
        }
        else
        {
            insert(point, node->right);
        }
    }
}

// Contains

bool BallTree::contains(const VectorXd& query_point, BallTreeNode::Ptr node, double tolerance) const {
    if (!node) {
        return false;
    }

    double dist_to_center = (query_point - node->center).norm();
    if (dist_to_center - node->radius > tolerance) {
        return false;
    }

    for (int i = 0; i < node->data.rows(); i++) {
        double dist = (query_point - node->data.row(i).transpose()).norm();
        if (dist <= tolerance) {
            return true;
        }
    }

    bool left_contains = node->left ? contains(query_point, node->left, tolerance) : false;
    bool right_contains = node->right ? contains(query_point, node->right, tolerance) : false;
    return left_contains || right_contains;
}

bool BallTree::contains(const VectorXd& query_point, double tolerance) const {
    return contains(query_point, root, tolerance);
}

// Size

int BallTree::size() const {
    return getSize(root);
}

int BallTree::getSize(const BallTreeNode::Ptr& node) const {
    if (!node) {
        return 0;
    }

    int count = node->data.rows(); // Number of data points in the current node
    count += getSize(node->left); // Add the count from the left subtree
    count += getSize(node->right); // Add the count from the right subtree

    return count;
}