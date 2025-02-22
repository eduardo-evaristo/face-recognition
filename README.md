# Face Recognition App

This project is a simple face recognition application using Flask and the `face_recognition` library. It provides an API endpoint to compare a live image with a set of predefined images and determine if there is a match.

## Getting Started

### Prerequisites

- Docker
- Python 3.10.3

### Installation

1. Clone the repository:

   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Build the Docker image:

   ```sh
   docker build . -t <your_image_name>:<your_image_tag>
   ```

3. Run the Docker container:
   ```sh
   docker run -d -p 5000:5000 --network host --name <your_container_name> <your_image_name>:<your_image_tag>
   ```

### Usage

The application exposes a single endpoint:

- `POST /recognize`: Accepts a screenshot image and compares it with predefined images to check for a match.

#### Example Request

```sh
curl -X POST http://localhost:5000/recognize -F "screenshot=@path_to_your_image.jpg"
```

### Files

- `Dockerfile`: Defines the Docker image for the application.
- `requirements.txt`: Lists the Python dependencies.
- `server.py`: Contains the Flask application and face recognition logic.

### Notes

- The application currently compares the uploaded image with two predefined images (`./pic_of_me.jpeg` and `./not_me.jpg`). You can modify these paths as needed.
- The application uses a virtual environment within the Docker container to manage its dependencies.

### License

This project is licensed under the MIT License.
