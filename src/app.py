import requests
import argparse
import base64
import logging
import pickle
import uuid
from datetime import datetime
from imutils import paths

# from src.ai_predictor import facePredictor
from src.detectfaces_mtcnn.Configurations import get_logger
from src.training.softmax import SoftMax
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


import os
import cv2
import numpy as np
from flask import  request, render_template, jsonify
from keras.models import load_model
from mtcnn import MTCNN

# from src.ai_predictor.facePredictor import FacePredictor
# from src.detectfaces_mtcnn.Configurations import ConfigurationsPOJO
from src.insightface.deploy import face_model
from src.insightface.src.common import face_preprocess

# app = Flask(__name__)
# app = Blueprint('main', __name__)
# from src.ai_predictor.facePredictor import face_predictor_bp
# app.register_blueprint(face_predictor_bp, url_prefix='/after_upload')

from flask import Flask
app = Flask(__name__)
from flask_cors import CORS
CORS(app)


from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from werkzeug.security import generate_password_hash, check_password_hash

url = "mongodb+srv://ayushmitian904:pymongo@ayush0.yiui1qf.mongodb.net/?retryWrites=true&w=majority&appName=Ayush0"
# Create a new client and connect to the server
client = MongoClient(url, server_api=ServerApi('1'))


class FacePredictor:
    def __init__(self, logFileName):
        try:
            self.logFileName = logFileName
            self.image_size = '112,112'
            self.model_path = "./insightface/models/model-y1-test2/model,0"
            self.threshold = 1.24
            self.det = 0
            self.model_filename = '../src/com_in_ineuron_ai_sorting/model_data/mars-small128.pb'

            # Initialize detector
            self.detector = MTCNN()

            # Initialize faces embedding model
            self.embedding_model = face_model.FaceModel(self.image_size, self.model_path, self.threshold, self.det)

            self.embeddings_path = "./faceEmbeddingModels/embeddings.pickle"
            self.le_names_path = "./faceEmbeddingModels/le_names.pickle"
            self.le_reg_path = "./faceEmbeddingModels/le_reg.pickle"

            # Check if files exist
            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(f"File not found: {self.embeddings_path}")
            if not os.path.exists(self.le_names_path):
                raise FileNotFoundError(f"File not found: {self.le_names_path}")
            if not os.path.exists(self.le_reg_path):
                raise FileNotFoundError(f"File not found: {self.le_reg_path}")

            # Load embeddings and labels
            self.data = pickle.loads(open(self.embeddings_path, "rb").read())
            self.le_names = pickle.loads(open(self.le_names_path, "rb").read())
            self.le_reg = pickle.loads(open(self.le_reg_path, "rb").read())

            self.embeddings = np.array(self.data['embeddings'])
            self.labels_names = self.le_names.transform(self.data['names'])
            self.labels_reg = self.le_reg.transform(self.data['registration_no'])

            # # Load the classifier model
            # self.model = load_model(ConfigurationsPOJO.clssfr_ModelPath)

            # Load the classifier models
            self.names_model_path = 'faceEmbeddingModels/names_model.h5'
            self.reg_model_path = 'faceEmbeddingModels/reg_model.h5'
            print("S4")
            self.names_model = load_model(self.names_model_path)
            print("S5")
            self.reg_model = load_model(self.reg_model_path)
            print(f"Names model loaded successfully: {self.names_model}")
            print(f"Registration model loaded successfully: {self.reg_model}")

            # Set up logging
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=self.logFileName,
                                level=logging.INFO,
                                datefmt='%Y-%m-%d %H:%M:%S')

        except Exception as e:
            print(e)

    # Define distance function
    @staticmethod
    def findCosineDistance(vector1, vector2):

        # Calculate cosine distance between two vector

        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def CosineSimilarity(test_vec, source_vecs):
        # Verify the similarity of one vector to group vectors of one class

        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += FacePredictor.findCosineDistance(test_vec, source_vec)
        return cos_dist / len(source_vecs)

    def detectFace(self, filepath):

        cosine_threshold = 0.8
        proba_threshold = 0.85
        comparing_num = 5

        '''Choose font and position'''
        font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        font_color = (255, 255, 255)  # White color
        thickness = 2
        fixed_face_size = (100, 100)

        json_data = []
        image_path = filepath
        print(image_path)
        frame = cv2.imread(image_path)
        if frame is None:
            print("No image captured.")
            return
        # print("entering 4")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = self.detector.detect_faces(rgb)
        # print("entering 5")

        print(len(bboxes))
        if len(bboxes) != 0:
            for bboxe in bboxes:
                face_matching_accuracy = None
                bbox = bboxe
                print(bbox)

                landmarks = bboxe['keypoints']
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                      landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                      landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                      landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2, 5)).T
                nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))
                embedding = self.embedding_model.get_feature(nimg).reshape(1, -1)
                # print(f"this is the embeddings\n\n{embedding}\n")

                name = "Unknown"
                reg_no = "Unknown"

                # Predict name class
                name_preds = self.names_model.predict(embedding)
                name_preds = name_preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(name_preds)
                proba = name_preds[j]
                # Compare this vector to source class vectors to verify it belongs to this class
                match_class_idx = (self.labels_names == j)
                match_class_idx = np.where(match_class_idx)[0]
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                # print(f"this is the selected_idx\n\n{selected_idx}\n")
                compare_embeddings = self.embeddings[selected_idx]
                cos_similarity = self.CosineSimilarity(embedding, compare_embeddings)
                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    # print(f"this is the j\n\n{j}\n")
                    # print(f"this is the self.le_names.classes_\n\n{self.le_names.classes_}\n")
                    name = os.path.basename(self.le_names.classes_[j])
                    face_matching_accuracy = f"{(proba * 100):.2f}"
                    print("Recognized: {} , {}".format(name, face_matching_accuracy))
                # print("Names encoded properly")

                # Predict registration number class
                reg_preds = self.reg_model.predict(embedding)
                reg_preds = reg_preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(reg_preds)
                proba_reg = reg_preds[j]
                # Compare this vector to source class vectors to verify it belongs to this class
                match_class_idx_reg = (self.labels_reg == j)
                match_class_idx_reg = np.where(match_class_idx_reg)[0]
                selected_idx_reg = np.random.choice(match_class_idx_reg, comparing_num)
                compare_embeddings_reg = self.embeddings[selected_idx_reg]
                cos_similarity_reg = self.CosineSimilarity(embedding, compare_embeddings_reg)
                if cos_similarity_reg < cosine_threshold and proba_reg > proba_threshold:
                    reg_no = self.le_reg.classes_[j]
                    print(reg_no)
                # print("Registration numbers encoded properly")

                if name == "Unknown":
                    face_matching_accuracy = "00.00"

                '''cropping the faces of the person predicted by the detector'''
                x, y, w, h = bbox['box']
                cropped_faces = frame[y - 5:y + h + 30, x - 10: x + w + 10]
                cropped_faces = cv2.resize(cropped_faces, fixed_face_size)

                font_scale = (min(cropped_faces.shape[0], cropped_faces.shape[1]) / 100.0)

                text_size = cv2.getTextSize(face_matching_accuracy, font, font_scale, thickness)[0]
                text_x = (cropped_faces.shape[1] - text_size[0]) // 2
                text_y = cropped_faces.shape[0] - 2  # Adjust 20 according to your preference

                position = (text_x, text_y)  # Position of the text (x, y)

                background_x1 = text_x - 5
                background_y1 = text_y - text_size[1] - 5
                background_x2 = text_x + text_size[0] + 5
                background_y2 = text_y + 5
                cv2.rectangle(cropped_faces, (background_x1, background_y1), (background_x2, background_y2),
                              (0, 0, 0), cv2.FILLED)
                cv2.putText(cropped_faces, face_matching_accuracy, position, font, font_scale, font_color,
                            thickness)

                retval, buffer = cv2.imencode('.jpg', cropped_faces)
                cropped_face_encoded_bytes = base64.b64encode(buffer)
                cropped_face_encoded_b64 = cropped_face_encoded_bytes.decode('utf-8')

                json_data.append({
                    "name": name,
                    "reg_no": reg_no,
                    "Confidence_Score": face_matching_accuracy,
                    "Cropped_image_encoded": cropped_face_encoded_b64
                })

            print(json_data)
            return json_data


class TrainingDataCollector:

    def __init__(self, args):
        self.args = args
        # Detector = mtcnn_detector
        self.detector = MTCNN()

    def collection_decider(self):
        registration_no = self.args["reg_no"]
        # extracting college and course code from reg no
        batch_code = int(registration_no[:2])  # 21
        course_code = registration_no[2:5]  # 106
        college_code = registration_no[5:8]  # 107
        regular_or_LE_code = registration_no[8:9]  # 9
        roll_code = registration_no[9:11]  # 04

        batch = None
        if regular_or_LE_code == "9":
            batch = "2K" + str(batch_code - 1)
        elif regular_or_LE_code == "0":
            batch = "2K" + str(batch_code)

        regular_or_LE_dic = {
            "0": "REG",
            "9": "LE"
        }

        course_dic = {
            "101": "CE",
            "102": "ME",
            "103": "EE",
            "104": "ECE",
            "105": "CSE",
            "106": "IT",
            "107": "LT",
            "108": "BArch",
            "109": "BPharm",
            "110": "EEE",
            "111": "IE"
        }

        college_dic = {
            "102": "VVIT_Purnia",
            "103": "NSIT_Patna",
            "106": "SITYOGIT_Aurangabad",
            "107": "MIT_Muzaffarpur",
            "108": "BCE_Bhagalpur",
            "109": "NCE_Nalanda",
            "110": "GEC_Gaya",
            "111": "DCE_Darbhanga",
            "113": "MCE_Mothihari",
            "115": "AIT_Kishanganj",
            "117": "LNJPIT_Chhapra",
            "118": "BIT_Gaya",
            "119": "AMIT_Banka",
            "121": "MBIT_Forbesganj",
            "122": "ECET_Vaishali",
            "123": "SETI_Siwan",
            "124": "SEC_Sasaram",
            "125": "RRSDCE_Begusarai",
            "126": "BCE_Patna",
            "127": "SIT_Sitamarhi",
            "128": "BPMCEM_Madhepura",
            "129": "KEC_Katihar",
            "130": "SCE_Supaul",
            "131": "PCE_Purnea",
            "132": "SCE_Saharsa",
            "133": "GEC_Jamui",
            "134": "GEC_Banka",
            "135": "GEC_Vaishali",
            "136": "MIT_Bihta",
            "139": "RPSIT_Patna",
            "140": "MACET_Patna",
            "141": "GEC_Nawada",
            "142": "GEC_Kishanganj",
            "144": "GEC_Munger",
            "145": "GEC_Sheohar",
            "146": "GEC_WestChamparan",
            "147": "GEC_Aurangabad",
            "148": "GEC_Kaimur",
            "149": "GEC_Gopalganj",
            "150": "GEC_Madhubani",
            "151": "GEC_Siwan",
            "152": "GEC_Jehanabad",
            "153": "GEC_Arwal",
            "154": "GEC_Khagaria",
            "155": "GEC_Buxar",
            "156": "GEC_Bhojpur",
            "157": "GEC_Sheikhpura",
            "158": "GEC_Lakhisarai",
            "159": "GEC_Samastipur",
            "165": "SPNREC_Araria"
        }

        collection = f"{college_dic[college_code]}_{course_dic[course_code]}_{batch}"

        return collection

    def collectImagesFromCamera(self):
        # initialize video stream
        cap = cv2.VideoCapture(0)

        # Setup some useful var
        faces = 0
        frames = 0
        max_faces = int(self.args["faces"])
        max_bbox = np.zeros(4)

        if not (os.path.exists(self.args["output"])):
            os.makedirs(self.args["output"])

        while faces < max_faces:
            ret, frame = cap.read()
            frames += 1

            dtString = str(datetime.now().microsecond)
            # Get all faces on current frame
            bboxes = self.detector.detect_faces(frame)

            if len(bboxes) != 0:
                # Get only the biggest face
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe["box"]
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    keypoints = bboxe["keypoints"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_bbox = bbox
                        landmarks = keypoints
                        max_area = area

                max_bbox = max_bbox[0:4]

                # get each of 3 frames
                if frames % 3 == 0:
                    # convert to face_preprocess.preprocess input
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(self.args["output"], "{}.jpg".format(dtString)), nimg)

                    # # Convert image data to binary
                    # binary_data = io.BytesIO(nimg).read()
                    # image_data = base64.b64encode(binary_data).decode('utf-8')

                    # Convert image data to binary
                    _, buffer = cv2.imencode('.jpg', nimg)
                    binary_data = buffer.tobytes()
                    image_data = base64.b64encode(binary_data).decode('utf-8')

                    #  sending images to database
                    db_flag = True
                    while db_flag:
                        reg_dbname = "Student_Registration_Database"
                        # reg_collectionName = "MIT_Muzaffarpur"
                        reg_collectionName = self.collection_decider()
                        # Access a database
                        db = client[reg_dbname]
                        # print(db)

                        # Access a collection
                        collection = db[reg_collectionName]

                        # Check if the document already exists
                        query = {'name': self.args['name'], 'reg_no': self.args['reg_no']}
                        existing_document = collection.find_one(query)

                        if existing_document:
                            # Update the existing document with the new image
                            update_query = {'$set': {f'image.{dtString}': image_data}}
                            collection.update_one(query, update_query)
                        else:
                            document = {
                                'name': self.args['name'],
                                'reg_no': self.args['reg_no'],
                                'image': {f'{dtString}': image_data}
                            }
                            collection.insert_one(document)
                            db_flag = False

                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class GenerateFaceEmbedding:

    def __init__(self, args):
        self.args = args
        self.image_size = '112,112'
        self.model = "./insightface/models/model-y1-test2/model,0"
        self.threshold = 1.24
        self.det = 0

    def genFaceEmbedding(self):
        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.args.dataset))
        # imagePaths = list(paths.list_images(f"{self.args.dataset}"))
        # print(imagePaths)

        # Initialize the faces embedder
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

        # Initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []
        knownRegistrations = []

        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

            # Extract the base directory and split to get the name and registration number
            dir_name = os.path.basename(os.path.dirname(imagePath))
            file_name = os.path.basename(dir_name).split('.')[0]  # Remove file extension

            # Assume the file name is in the format 'name reg_no'
            name_parts = file_name.rsplit(' ', 1)

            if len(name_parts) == 2:
                name, registration_no = name_parts
            else:
                print(f"[WARN] Unable to parse name and registration number from {imagePath}")
                continue

            # load the image
            image = cv2.imread(imagePath)
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face
            # embedding to their respective list
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            knownRegistrations.append(registration_no)
            total += 1

        print(total, " faces embedded")

        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames, "registration_no": knownRegistrations}

        f = open(self.args.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()

        # f = open(self.args.embeddings, "wb")
        # f.write(pickle.dumps(data))
        # f.close()


class TrainFaceRecogModel:

    def __init__(self, args):
        self.args = args
        self.logger = get_logger()
        # Load the face embeddings
        self.data = pickle.loads(open(args.embedding, "rb").read())

    def trainKerasModelForFaceRecognition(self):

        # ------------------------------------------------------------------------------------
        # # Encode the labels
        # le = LabelEncoder()
        # labels = le.fit_transform(self.data["names"])
        # num_classes = len(np.unique(labels))
        # labels = labels.reshape(-1, 1)
        # one_hot_encoder = OneHotEncoder(categorical_features=[0])
        # labels = one_hot_encoder.fit_transform(labels).toarray()
        # # print(self.data)
        # --------------------------------------------------------------------------------------------------

        # Encode the name labels
        le_names = LabelEncoder()
        labels_names = le_names.fit_transform(self.data["names"])
        num_classes_names = len(np.unique(labels_names))
        labels_names = labels_names.reshape(-1, 1)
        one_hot_encoder_names = OneHotEncoder(categories='auto')
        labels_names = one_hot_encoder_names.fit_transform(labels_names).toarray()

        # Encode the registration number labels
        le_reg = LabelEncoder()
        labels_reg = le_reg.fit_transform(self.data["registration_no"])
        num_classes_reg = len(np.unique(labels_reg))
        labels_reg = labels_reg.reshape(-1, 1)
        one_hot_encoder_reg = OneHotEncoder(categories='auto')
        labels_reg = one_hot_encoder_reg.fit_transform(labels_reg).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        # Build softmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes_names)
        model_names = softmax.build()

        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes_reg)
        model_reg = softmax.build()

        # Create KFold
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history_names = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
        history_reg = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Train the model for names
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train_names, y_val_names = embeddings[train_idx], embeddings[valid_idx], labels_names[
                train_idx], labels_names[valid_idx]
            his_names = model_names.fit(X_train, y_train_names, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                                        validation_data=(X_val, y_val_names))
            history_names['acc'] += his_names.history['acc']
            history_names['val_acc'] += his_names.history['val_acc']
            history_names['loss'] += his_names.history['loss']
            history_names['val_loss'] += his_names.history['val_loss']
            self.logger.info(his_names.history['acc'])

        # Train the model for registration numbers
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train_reg, y_val_reg = embeddings[train_idx], embeddings[valid_idx], labels_reg[
                train_idx], labels_reg[valid_idx]
            his_reg = model_reg.fit(X_train, y_train_reg, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                                    validation_data=(X_val, y_val_reg))
            history_reg['acc'] += his_reg.history['acc']
            history_reg['val_acc'] += his_reg.history['val_acc']
            history_reg['loss'] += his_reg.history['loss']
            history_reg['val_loss'] += his_reg.history['val_loss']
            self.logger.info(his_reg.history['acc'])

        # Save the face recognition models
        model_names.save(self.args.model_names)
        model_reg.save(self.args.model_reg)

        # Pickle the label encoders
        with open(self.args.le_names, "wb") as f:
            pickle.dump(le_names, f)

        with open(self.args.le_reg, "wb") as f:
            pickle.dump(le_reg, f)


@app.route("/train", methods=["POST"])
def train_model():
    ap = argparse.ArgumentParser()

    # ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle",
                    help="path to serialized db of facial embeddings")
    ap.add_argument("--model_names", default="faceEmbeddingModels/names_model.h5",
                    help="path to output trained model for names")
    ap.add_argument("--model_reg", default="faceEmbeddingModels/reg_model.h5",
                    help="path to output trained model for registration numbers")
    ap.add_argument("--le_names", default="faceEmbeddingModels/le_names.pickle",
                    help="path to output label encoder for names")
    ap.add_argument("--le_reg", default="faceEmbeddingModels/le_reg.pickle",
                    help="path to output label encoder for registration numbers")

    # ap.add_argument("--model", default="faceEmbeddingModels/my_model.h5",
    #                 help="path to output trained model")
    # ap.add_argument("--le", default="faceEmbeddingModels/le.pickle",
    #                 help="path to output label encoder")

    ap.add_argument("--dataset", default="../datasets/train",
                    help="Path to training dataset")
    ap.add_argument("--embedding", default="faceEmbeddingModels/embeddings.pickle")
    # Argument of insightface
    ap.add_argument('--image-size', default='112,112', help='')
    ap.add_argument('--models', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
    ap.add_argument('--ga-model', default='', help='path to load model.')
    ap.add_argument('--gpu', default=0, type=int, help='gpu id')
    ap.add_argument('--det', default=0, type=int,
                    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

    args = ap.parse_args()

    genFaceEmbdng = GenerateFaceEmbedding(args)
    genFaceEmbdng.genFaceEmbedding()
    # genFaceEmbdng.load_and_inspect_embeddings()

    faceRecogModel = TrainFaceRecogModel(args)
    faceRecogModel.trainKerasModelForFaceRecognition()

    notifctn = "Model training is successful.No you can go for prediction."
    return jsonify({"message": notifctn})


@app.route("/registration", methods=["POST"])
def register_user():
    name = request.form.get('name')
    registration_no = request.form.get('registration_no')
    file_name = name + " " + registration_no
    # image_file = request.files.get('image')

    if not (name and registration_no):
        return jsonify({'error': 'Name and Reg no are required.'}), 400

    ap = argparse.ArgumentParser()

    ap.add_argument("--faces", default=50,
                    help="Number of faces that camera will get")
    ap.add_argument("--output", default="../datasets/train/" + file_name,
                    help="Path to faces output")
    ap.add_argument("--reg_no", default=registration_no,
                    help="Path to faces output")
    ap.add_argument("--name", default=name,
                    help="Path to faces output")

    args = vars(ap.parse_args())
    # args = (ap.parse_args())
    # print(args)
    trnngDataCollctrObj = TrainingDataCollector(args)
    # print("object created")
    trnngDataCollctrObj.collectImagesFromCamera()
    return jsonify({"message": "User Registration Done"})


@app.route("/predict", methods=["POST", "GET"])
def predict():
    file = request.files['image']
    print("file received")

    if not file:
        return jsonify({'error': 'No file part'})

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # base 64 making of image
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({'error': 'Failed to read the image file'}), 500
    base64_image_str = base64.b64encode(image_bytes).decode('utf-8')

    #     making directory for the new files
    UPLOAD_FOLDER = 'Image_to_be_predicted'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    #        naming files
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    # Save the file
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    logFileName = "ProceduralLog.txt"
    class_obj = FacePredictor(logFileName)
    json_data = class_obj.detectFace(filepath)

    # checking whether all the reg no batch matches
    batch_dic = []
    reg_no = None
    i = -1
    flag = True
    batch = None
    for each in json_data:
        if each['reg_no'] != "Unknown":
            reg_no = each['reg_no']
        batch_no = each['reg_no'][:2]
        regular_or_LE_code = each['reg_no'][8:9]
        if regular_or_LE_code == 9:
            batch = "2K" + str(batch_no - 1)
        elif regular_or_LE_code == 0:
            batch = "2K" + str(batch_no)
        batch_dic.append(batch)
        if batch_dic[++i] == batch:
            pass
            # print(flag)
        else:
            flag = False
            break
    print(f"This is Reg No. {reg_no}")
    if flag:
        ap = argparse.ArgumentParser()
        ap.add_argument("--reg_no", default=reg_no,
                        help="Path to faces output")
        args = vars(ap.parse_args())
        trng_class_obj = TrainingDataCollector(args)

        # Database storaging
        attendance_collectionName = trng_class_obj.collection_decider()
        # attendance_collectionName = "MIT_Muzaffarpur"
        attendance_dbname = "Student_Attendance_Database"
        # Access a database
        db = client[attendance_dbname]

        # Access a collection
        collection = db[attendance_collectionName]

        current_date = datetime.now().date().isoformat()

        document = {
            "Date": current_date,
            "Students_present": json_data,
            "Class_Photo": base64_image_str
        }
        collection.insert_one(document)
        return jsonify(json_data)
    else:
        return "Students of Different branches present."

# api to retrieve the whole class attendance
@app.route("/retrieve_whole_class_attendance", methods=['GET', 'POST'])
def retrieve_attendance():
    college = request.form.get('college')
    course = request.form.get('course')
    batch = request.form.get('batch')
    print(college, course, batch)

    attendance_collectionName = f"{college}_{course}_{batch}"
    print(f"This is the collection name : {attendance_collectionName}")

    # Access a database
    reg_dbname = "Student_Registration_Database"
    reg_db = client[reg_dbname]

    # Access a collection
    reg_collection = reg_db[attendance_collectionName]
    stud_pres_det = reg_collection.find({})
    # print(stud_pres_det)
    response_data = []
    for doc in stud_pres_det:
        reg_no = (doc['reg_no'])
        if reg_no == "Unknown":
            continue
        else:
            # attendance_db_name = "Student_Attendance_Database"
            # attendance_db = client[attendance_db_name]
            # attendance_coll = attendance_db[attendance_collectionName]
            request_data = {'reg_no': reg_no,
                            'attendance_collection': attendance_collectionName,
                            'token': True}
            api_response = requests.post('http://localhost:5000/retrieve_image_uploaded_reg', data=request_data)

            # Process the response from the external API if needed
            processed_text = api_response.json()[0]
            response_data.append(processed_text)
    print(response_data)
    return jsonify(response_data)


# endpoint to retrieve the registration details using Registration Number
@app.route("/retrieve_image_uploaded_date", methods=['GET', 'POST'])
def retrieve_by_date():
    date = request.form.get('date')
    college = request.form.get('college')
    course = request.form.get('course')
    batch = request.form.get('batch')
    print(date, college, course, batch)

    attendance_collectionName = f"{college}_{course}_{batch}"
    print(f"This is the collection name : {attendance_collectionName}")

    # Access a database
    attendance_dbname = "Student_Attendance_Database"
    db = client[attendance_dbname]

    # Access a collection
    collection = db[attendance_collectionName]
    query = {
        'Date': date
    }
    stud_pres_det = collection.find_one(query)["Students_present"]
    # print(stud_pres_det)

    response_data = []
    for each_coll in stud_pres_det:
        reg_no = each_coll['reg_no']
        if reg_no == "Unknown":
            continue
        else:
            request_data = {'reg_no': reg_no,
                            'attendance_collection': attendance_collectionName,
                            'token': True}
            api_response = requests.post('http://localhost:5000/retrieve_image_uploaded_reg', data=request_data)

            # Process the response from the external API if needed
            processed_text = api_response.json()[0]
            response_data.append(processed_text)

    print(response_data)
    return jsonify([response_data])


# endpoint to retrieve the registration details using Registration Number
@app.route("/retrieve_image_uploaded_reg", methods=['GET', 'POST'])
def retrieve_by_reg():
    reg_no = request.form.get('reg_no')
    token = request.form.get('token', False)  # to check its source
    attendance_collectionName = request.form.get('attendance_collection', " ")
    print(reg_no)

    if not token:
        ap = argparse.ArgumentParser()
        ap.add_argument("--reg_no", default=reg_no,
                        help="Path to faces output")
        args = vars(ap.parse_args())
        trng_class_obj = TrainingDataCollector(args)
        # Database storaging
        attendance_collectionName = trng_class_obj.collection_decider()

    attendance_dbname = "Student_Attendance_Database"
    # Access a database
    db = client[attendance_dbname]

    # Access a collection
    collection = db[attendance_collectionName]
    documents = collection.find({})
    # print(documents)
    pres_dates = []
    total_no_of_days = []
    name = None
    cropped_img = None
    for doc in documents:
        total_no_of_days.append(doc.get('Date'))
        for each in doc['Students_present']:
            if reg_no == each['reg_no']:
                if name is None or cropped_img is None:
                    name = each['name']
                    cropped_img = each['Cropped_image_encoded']
                pres_dates.append(doc['Date'])
    pres_dates = set(pres_dates)
    total_no_of_days = set(total_no_of_days)
    present_percentage = ((len(pres_dates) * 100) / len(total_no_of_days))

    response_data = {"No_of_Pres_dates": len(pres_dates),
                     "Present_percentage": present_percentage,
                     "total_no_of_days": len(total_no_of_days),
                     "reg_no": reg_no,
                     "name": name,
                     "Cropped_image_encoded": cropped_img}

    # print(response_data)
    return jsonify([response_data])



@app.route("/login_registration", methods=['POST'])
def login_register():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    email = request.form.get('email')
    mobile = request.form.get('mobile')
    username = request.form.get('username')
    password = request.form.get('password')
    hashed_password = generate_password_hash(password)

    login_dbname = "Login_Registration_Database"
    login_collection_name = "Login_Details"
    db = client[login_dbname]
    collection = db[login_collection_name]
    collection.insert_one({
        "username": username,
        "password": hashed_password,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "mobile": mobile
    })
    return jsonify({"message": "Registration successful"}), 200


@app.route("/login_sign_in", methods=['POST'])
def login_signin():
    # Get username and password from request
    username = request.form.get('username')
    password = request.form.get('password')
    print("this is the username and password")
    print(username , password)
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    login_dbname = "Login_Registration_Database"
    login_collection_name = "Login_Details"
    db = client[login_dbname]
    collection = db[login_collection_name]
    # Find the user in the MongoDB database
    user = collection.find_one({"username": username})

    print("this is the last section")
    if user:
        # Check if the password matches
        if check_password_hash(user["password"], password):
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Invalid password"}), 401
    else:
        return jsonify({"error": "User not found"}), 404


@app.route("/registration_start")
def registration_start():
    return render_template('registration.html')


@app.route("/train_start")
def train_start():
    return render_template('embeddings.html')


@app.route("/predict_start")
def predict_start():
    return render_template('prediction.html')


@app.route("/retrieve_start")
def retrieve_start():
    return render_template("retrieve_attendance_by_date.html")


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

# image better without confidence score
# reg unknown removal
# endpoint for whole students of a batch attendance
# implement integrity constraint on image capturing while training images
# introduce timing of the class during attendance marking