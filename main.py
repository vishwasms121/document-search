from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
# from flask_socketio import SocketIO, emit
from datetime import datetime
import config.config as cfg
from models import AllFiles, MasterExtraction, db
import time
from werkzeug.utils import secure_filename
import os
from functools import wraps
import json
from copy import deepcopy
from waitress import serve
from nlp import *
import binascii
import pandas as pd
from copy import deepcopy
import threading
from flasgger import Swagger


UPLOAD_FOLDER = 'public/docs'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__, static_url_path='')

print(cfg.DB_URI)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["SECRET_KEY"] = "vcEmFS5PWG"
app.config["SQLALCHEMY_DATABASE_URI"] = cfg.DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = 'public/docs'

db.init_app(app)
CORS(app)
Swagger(app, template_file='config/swagger.yml')
max_upload_limit = 100


def generate_custom_id():
    return (
        binascii.b2a_hex(os.urandom(4)).decode()
        + hex(int(time.time() * 10**5) % 10**12)[2:]
    )


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/search/sync', methods=['GET'])
@cross_origin()
def sync():
    def background_extract(**kwargs):
        file_arr = kwargs.get('post_data', [])
        output = extract_convert_text_from_pdf_to_json(file_arr)
        out = list(output)
        for ele in out:
            obj = json.loads(ele)
            with app.app_context():
                file_id = AllFiles.query.filter_by(
                    filename=obj["filename"]).first().id
                for matter in list(obj["content"]):
                    db_insert = MasterExtraction(file_id, matter['page_number'], matter['para_number'],
                                                 matter['d'], matter['cleaned_d'], datetime.now())
                    db.session.add(db_insert)
                db.session.commit()
                db.session.remove()
        print('extraction completed')

    only_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(
        os.path.join(app.config['UPLOAD_FOLDER'], f))]
    file_array = []
    for file in only_files:
        check_file_exists = AllFiles.query.filter_by(
            filename=file).first()
        if not check_file_exists:
            file_array.append(file)
            file_insert = AllFiles(file, datetime.now())
            db.session.add(file_insert)
            db.session.commit()
    t = threading.Thread(target=background_extract, kwargs={
        'post_data': file_array})
    t.start()
    print('main thread completed')
    return jsonify({"message": "Success"})


@app.route('/search/uploadFiles', methods=['POST'])
@cross_origin()
def uploadFiles():
    file_type = request.form.get('type')
    file_arr = []
    file_upload_state = False
    if file_type == 'multi':
        if request.files.get('files'):
            files = request.files.getlist('files')
            if len(files) < max_upload_limit:
                for file in files:
                    if file and allowed_file(file.filename):
                        check_file_exists = AllFiles.query.filter_by(
                            filename=file.filename).first()
                        if not check_file_exists:
                            filename_secure = secure_filename(file.filename)
                            path = os.path.join(
                                app.config['UPLOAD_FOLDER'], str(
                                    filename_secure)
                            )
                            file.save(path)
                            file_arr.append(filename_secure)
                            file_data = AllFiles(
                                filename_secure, datetime.now())
                            db.session.add(file_data)
                            file_upload_state = True
                if file_upload_state:
                    db.session.commit()
                    db.session.remove()
                    output = extract_convert_text_from_pdf_to_json(file_arr)
                    out = list(output)
                    for ele in out:
                        obj = json.loads(ele)
                        file_id = AllFiles.query.filter_by(
                            filename=obj["filename"]).first().id
                        for matter in list(obj["content"]):
                            db_insert = MasterExtraction(file_id, matter['page_number'], matter['para_number'],
                                                         matter['d'], matter['cleaned_d'], datetime.now())
                            db.session.add(db_insert)
                        db.session.commit()
                        db.session.remove()
                    return jsonify({"numFilesUploaded": len(file_arr)})
                else:
                    return jsonify({"message": "All Files Already Uploaded"})
            else:
                return jsonify({"message": "Please reduce number of files"}), 400
        else:
            return jsonify({"message": "Missing Params"}), 400
    else:
        return jsonify({"message": "Work in progress"})


@app.route('/search/predict', methods=['POST'])
@cross_origin()
def predict():
    user_phrase = request.json.get('question')
    join_data = (db.session.query(MasterExtraction, AllFiles)).join(
        AllFiles, MasterExtraction.file_id == AllFiles.id).all()
    all_data = []
    for ele in join_data:
        d = {}
        d['pdf_name'] = ele[1].filename
        d['page'] = ele[0].page_num
        d['documents'] = ele[0].extracted_text
        d['documents_cleaned'] = ele[0].cleaned_extracted_text
        deep_d = deepcopy(d)
        all_data.append(deep_d)

    df = pd.DataFrame(all_data)
    tfidfvectoriser, tfidf_vectors = similarity_model(df)
    pairwise_similarities = similarity(
        tfidfvectoriser, tfidf_vectors, user_phrase)
    df['similarities'] = pairwise_similarities
    final_df = df.sort_values(by=['similarities'], ascending=False)
    result = final_df.head(10)
    json_list = json.loads(json.dumps(list(result.T.to_dict().values())))
    output = []
    for res in json_list:
        di = {}
        di['pdfName'] = res['pdf_name']
        di['answer'] = res['documents']
        di['pageNumber'] = res['page']
        deep_di = deepcopy(di)
        output.append(deep_di)
    return jsonify(output)


if __name__ == "__main__":
    print("starting...")
    serve(app, host=cfg.Flask["HOST"],
          port=cfg.Flask["PORT"])
    # app.run(
    #     host=cfg.Flask["HOST"],
    #     port=cfg.Flask["PORT"],
    #     threaded=cfg.Flask["THREADED"],
    #     debug=True,
    # )
