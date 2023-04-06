#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

class BallTreeNode {
public:
    using Ptr = std::shared_ptr<BallTreeNode>;
    Eigen::MatrixXd data;
    Eigen::VectorXd center;
    double radius;
    Ptr left;
    Ptr right;

    BallTreeNode(const Eigen::MatrixXd& data);
};

class BallTree {
public:
    using Ptr = std::shared_ptr<BallTree>;
    BallTreeNode::Ptr root;

    explicit BallTree(const Eigen::MatrixXd& data);

    std::vector<Eigen::VectorXd> kNearestNeighbors(const Eigen::VectorXd& query_point, int k) const;

private:
    BallTreeNode::Ptr buildBallTree(const Eigen::MatrixXd& data);
    std::vector<Eigen::MatrixXd> splitByDimension(const Eigen::MatrixXd& data, int dim, double val);
    static Eigen::VectorXi indicesToSlice(const std::vector<int>& indices);

    void searchKNearestNeighbors(const Eigen::VectorXd& query_point, int k, BallTreeNode::Ptr node, std::vector<std::pair<double, Eigen::VectorXd>>& neighbors) const;
};