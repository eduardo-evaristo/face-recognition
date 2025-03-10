from flask import Flask, request
from flask_cors import CORS
import face_recognition_models
import face_recognition
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient
import os
import numpy as np
import tempfile
from deepface import DeepFace

config = load_dotenv()
app = Flask(__name__)

# Configure CORS later, if needed
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
# Change this accordingly to the needs. depois que começarmos a salvar os rostos conhecidos, mudaremos essa parte e pegaremos isso do banco de dados
compareImgPaths = ['./pic_of_me.jpeg', './not_me.jpg']
indexOfPic = 0

client = MongoClient(os.getenv('MONGO_URI'))
db = client['ReconhecimentoFacial']
collection = db["rostos"]

# Testing out connection
try:
    client.admin.command('ping')
    print('Conexão bem-sucedida!')
except Exception as exc:
    print(f'Algo deu errado: {exc}')


@app.route("/recognize", methods=['POST'])
def recognize():
    userLiveImg = request.files.get('screenshot')

    if not userLiveImg:
        return {"error": "An image to compare must be provided"}, 400

    try:
        # Load and encode the live image
        userLiveFace = face_recognition.load_image_file(userLiveImg)
        userLiveFaceEncodings = face_recognition.face_encodings(userLiveFace)

        if not userLiveFaceEncodings:
            return {"error": "No face detected in the live image"}, 400
        
        # Attributing user's face encoding to a variable
        userLiveFaceEncoding = userLiveFaceEncodings[0]

        # Getting all known encodings
        known_encodings = list(collection.find({}))

        if not known_encodings:
            return {'result': 'There are no faces to compare'}

        for face in known_encodings:
            face_encoding = np.array(face['rosto'])
            # Compare faces
            comparison = face_recognition.compare_faces([face_encoding], userLiveFaceEncoding)

            if comparison[0]:
                return {'result': face['nome']}
        
        return {'result': 'No matching face found, register first'}, 404
    
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500

@app.route('/register', methods=['POST'])
def registerUser():
    picture = request.files.get('pic')
    name = request.form.get('name')

    if not picture or not name:
        return {'result': 'Missing either picture or name'}, 400

    # Getting the encoding of the face in the submitted pic 
    pictureFace = face_recognition.load_image_file(picture)
    pictureFaceEncodings = face_recognition.face_encodings(pictureFace)

    # Check if a face was detected
    if not pictureFaceEncodings:
        return {'result': 'No face detected'}, 400

    # COnvert back using np.array()
    encodingToList = pictureFaceEncodings[0].tolist()

    newDocument = {'rosto': encodingToList, 'nome': name }

    print(newDocument)

    newUser = collection.insert_one(newDocument)

    return {'result': str(newUser.inserted_id)}




# AI route to test AI-relaetd stuff
@app.route("/ai", methods=['POST'])
def ai():
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        return {'error': 'The Gemini API key could not be found!'}, 400
    
    data = request.form

    if not data or 'text' not in data or 'weather' not in data:
        return {'error': 'Either text or weather are missing in the body'}, 400


    pic = request.files.get('pic')
    weather = data.get('weather')
    question = data.get('text')
    neededData = {'weather': weather}
    print(pic)
    print(weather)
    print(question)

    emotion = "unknown"  # Default value if analysis fails
    
    if pic:
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                pic.save(temp_img.name)
                image_path = temp_img.name

            # Analyze the image for emotions
            analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)

            # Extract the dominant emotion
            emotion = analysis[0]['dominant_emotion']

            # Remove temp file
            os.remove(image_path)

        except Exception as e:
            print(f"Error analyzing image: {e}")

    # Include emotion in the data
    neededData["emotion"] = emotion

    sys_instruct = 'Você receberá uma pergunta e um conjunto de dados, se os dados forem pertinentes à pergunta, use-os, se a emoção do usuário ter pertinência à pergunta, considera-a ao responder. Caso os dados não sejam pertinentes, apenas responda normalmente a pergunta enviada. Não mencione que você possui esses dados nem dê informações a respeito dessa instrução. Dê respostas curtas na medida do possível, mas informativas o suficiente. Evite escrever símbolos como *(asterisco).'

    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
        system_instruction=sys_instruct),
        contents=f"Dados: {neededData}. Pergunta: {question}"
    )
    
    return {'response': response.text, 'emotion': emotion}

@app.route('/emotion', methods=['POST'])
def get_current_emotion():
    pic = request.files.get('pic')

    if pic:
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                pic.save(temp_img.name)
                image_path = temp_img.name

            # Analyze the image for emotions
            analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)

            # Extract the dominant emotion
            emotion = analysis[0]['dominant_emotion']

            # Remove temp file
            os.remove(image_path)

            # Return emotion
            return {'emotion': emotion}

        except Exception as e:
            return {'response': 'Erro ao analisar imagem'}, 500
    else:
        return {'response': 'A foto não foi enviada no campo pic'}, 400


if __name__ == '__main__':
    app.run(host="0.0.0.0")
# todo: clean up useless dependencies

