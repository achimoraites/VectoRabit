# VectoRabit

This project aims to enable users to quickly find the most similar vectors to a given query vector within a large dataset. 
It utilizes the Eigen library for linear algebra operations and the Crow library to create a RESTful API.

![download](https://user-images.githubusercontent.com/4193340/232200944-8642110c-3922-4177-b092-7519e3817ec8.jpg)


## Key Features:
- **Nearest-neighbors search** 
- **RESTful API**: allows users to send vector data and query via HTTP requests.

## Upcoming Features:
The following are subject to change.
- **Extensible Query Schema**: The project will support an extensible query schema similar to Elasticsearch, allowing users to define custom queries and filter results based on additional fields associated with the vectors.
- **CRUD operations**: Users can insert, edit and delete the stored data. 
- **Documentation**: The project will include clear documentation and examples to help users integrate it into their applications quickly.
- **Speed Optimization**:  Fast query times even with large datasets.
- **Dockerization**: it will be available for use via a Docker Image


## Example Use Cases:
- Content-based image retrieval: Given a query image, find similar images in a large dataset based on feature vectors extracted from the photos.
- Document similarity
- Recommendation systems
- Anomaly detection
- Clustering and classification
