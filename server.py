from flask import Flask, request
from flask_cors import CORS
import face_recognition_models
import face_recognition


app = Flask(__name__)

# Configure CORS later, if needed
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
# Change this accordingly to the needs. depois que começarmos a salvar os rostos conhecidos, mudaremos essa parte e pegaremos isso do banco de dados
compareImgPaths = ['./pic_of_me.jpeg', './not_me.jpg']
indexOfPic = 0


@app.route("/recognize", methods=['POST'])
def recognize():
    global indexOfPic
    # compareImg = request.files.get('comparison')
    userLiveImg = request.files.get('screenshot')
    print(request.files)

    if not userLiveImg:
        return {"error": "An image to compare must be provided"}, 400

    try:
        # Load and encode the live image
        userLiveFace = face_recognition.load_image_file(userLiveImg)
        userLiveFaceEncodings = face_recognition.face_encodings(userLiveFace)
        print(userLiveFaceEncodings[0])
        if not userLiveFaceEncodings:
            return {"error": "No face detected in the live image"}, 400
        
        # Load and encode the comparison image
        compareFace = face_recognition.load_image_file(compareImgPaths[indexOfPic])
        indexOfPic = 1 if indexOfPic == 0 else 0
        compareFaceEncodings = face_recognition.face_encodings(compareFace)
        print(compareFaceEncodings[0])
        if not compareFaceEncodings:
            return {"error": "No face detected in the comparison image"}, 400

        compareFaceEncoding = compareFaceEncodings[0]


        userLiveFaceEncoding = userLiveFaceEncodings[0]

        # Compare faces
        results = face_recognition.compare_faces([compareFaceEncoding], userLiveFaceEncoding)
        if results[0] == True:
            string = "It's a picture of me!"
            print(string)
            return {'result': True}
        else:
            string = "It's not a picture of me!"
            print(string)
            return {'result': False}
    

    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500


if __name__ == '__main__':
    app.run(debug=True)
    
# todo: clean up useless dependencies

