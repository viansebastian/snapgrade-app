import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from main_pipelines import main_circles_pipeline

app = Flask(__name__)
CORS(app)

@app.route('/process-circles', methods=['POST'])
def process_circles():
    master_sheet_file = request.files['master_sheet']
    student_answer_file = request.files['student_answer']
    
    print(f"Received master_sheet: {master_sheet_file.filename}")
    print(f"Received student_answer: {student_answer_file.filename}")

    # convert received img to cv2 format
    master_sheet_data = np.frombuffer(master_sheet_file.read(), np.uint8)
    master_sheet = cv2.imdecode(master_sheet_data, cv2.IMREAD_COLOR)  

    student_answer_data = np.frombuffer(student_answer_file.read(), np.uint8)
    student_answer = cv2.imdecode(student_answer_data, cv2.IMREAD_COLOR)  

    # Check if the images were read correctly
    if master_sheet is None or student_answer is None:
        return jsonify({"error": "Failed to process the images."}), 400

    # processing pipeline
    stu_final_score, stu_answer_key, detected_total_questions, detected_mistakes = main_circles_pipeline(master_sheet, student_answer)

    # convert the result image (stu_answer_key) from OpenCV to PNG format
    # then encode to base64 string
    _, buffer = cv2.imencode('.png', stu_answer_key)
    img_io = buffer.tobytes()
    img_base64 = base64.b64encode(img_io).decode('utf-8')

    # response dictionary (json)
    response = {
        'score': stu_final_score,
        'total_questions': detected_total_questions,
        'mistakes': detected_mistakes,
        'student_answer_key': img_base64  
    }

    return jsonify(response)  

if __name__ == '__main__':
    app.run(debug=True, port=5000)