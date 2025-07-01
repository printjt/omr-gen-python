import base64
import io

import cv2
import imutils
import numpy as np
from flask import Flask, request, jsonify, send_file
from imutils import contours
from imutils.perspective import four_point_transform
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)


class OMRGenerator:
    def __init__(self):
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 40

    def create_omr_sheet(self, num_questions, choices, student_info=True, title="EXAM"):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(self.page_width / 2, self.page_height - 50, title)

        y_pos = self.page_height - 100
        if student_info:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(self.margin, y_pos, "NAME: ________________________")
            c.drawString(self.margin + 250, y_pos, "ID: ________________")
            y_pos -= 30
            c.drawString(self.margin, y_pos, "CLASS: _______________")
            c.drawString(self.margin + 250, y_pos, "DATE: ______________")
            y_pos -= 50

        c.setFont("Helvetica", 10)
        c.drawString(self.margin, y_pos, "Instructions: Fill the bubbles completely with a dark pencil or pen.")
        y_pos -= 30

        bubble_radius = 10
        bubble_spacing = 30
        question_spacing = 35
        questions_per_column = 15  # Changed from 25 to 15 questions per column
        column_spacing = 200  # Space between columns

        start_x = self.margin + 20
        start_y = y_pos
        current_x = start_x
        current_y = start_y

        for q in range(1, num_questions + 1):
            # If we've filled one column and need to start a new one
            if (q - 1) % questions_per_column == 0 and q > 1:
                current_x += column_spacing
                current_y = start_y

            c.setFont("Helvetica-Bold", 11)
            c.drawString(current_x, current_y, f"{q}.")
            choice_x = current_x + 40
            for i, choice in enumerate(choices):
                c.circle(choice_x + (i * bubble_spacing), current_y + 3, bubble_radius, fill=0)
                c.setFont("Helvetica", 10)
                c.drawCentredString(choice_x + (i * bubble_spacing), current_y - 1, choice)

            current_y -= question_spacing

            # If we're at the bottom of the page and have more questions, create a new page
            if current_y < 100 and q < num_questions:
                c.showPage()
                current_x = start_x
                current_y = self.page_height - 50  # Start near the top of the new page
                start_y = current_y  # Update the starting position for the new page

        c.setFont("Helvetica", 8)
        c.drawCentredString(self.page_width / 2, 30, "CREATED BY OMR GENERATOR SYSTEM")

        # Add corner markers (only on the first page)
        marker_size = 4
        c.drawImage("omr_marker.jpg", self.margin, self.page_height - self.margin, height=20, width=20)
        c.drawImage("omr_marker.jpg", self.page_width - self.margin, self.page_height - self.margin, height=20,
                    width=20)
        c.drawImage("omr_marker.jpg", self.margin, self.margin + 50, height=20, width=20)
        c.drawImage("omr_marker.jpg", self.page_width - self.margin, self.margin + 50, height=20, width=20)
        # c.circle(self.margin, self.page_height - self.margin, marker_size, fill=1)
        # c.circle(self.page_width - self.margin, self.page_height - self.margin, marker_size, fill=1)
        # c.circle(self.margin, self.margin + 50, marker_size, fill=1)
        # c.circle(self.page_width - self.margin, self.margin + 50, marker_size, fill=1)

        c.save()
        buffer.seek(0)
        return buffer


class OMRScanner:
    def __init__(self):
        self.answer_key = []
        self.num_questions = 5
        self.choices = ['A', 'B', 'C', 'D', 'E']

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        return gray, edged

    def find_document(self, edged, original):
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        document_contour = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                document_contour = approx
                break

        if document_contour is None:
            return original, cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        paper = four_point_transform(original, document_contour.reshape(4, 2))
        warped = four_point_transform(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), document_contour.reshape(4, 2))

        return paper, warped

    def find_bubbles(self, thresh):
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        bubble_contours = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if (w >= 15 and h >= 15 and w <= 50 and h <= 50 and aspect_ratio >= 0.7 and aspect_ratio <= 1.3):
                bubble_contours.append(c)

        return bubble_contours

    def process_omr_sheet(self, image):
        try:
            gray, edged = self.preprocess_image(image)
            paper, warped = self.find_document(edged, image)

            thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble_contours = self.find_bubbles(thresh)

            if len(bubble_contours) == 0:
                return {"error": "No bubbles detected"}

            bubble_contours = contours.sort_contours(bubble_contours, method="top-to-bottom")[0]
            correct_answers = 0
            student_answers = {}
            num_choices = len(self.choices)

            for (q, i) in enumerate(np.arange(0, len(bubble_contours), num_choices)):
                if i + num_choices > len(bubble_contours):
                    break

                cnts = contours.sort_contours(bubble_contours[i:i + num_choices])[0]
                bubbled = None

                for (j, c) in enumerate(cnts):
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(masked)

                    if bubbled is None or total > bubbled[0]:
                        bubbled = (total, j)

                color = (0, 0, 255)
                if q < len(self.answer_key):
                    k = self.answer_key[q]
                    if k == bubbled[1]:
                        color = (0, 255, 0)
                        correct_answers += 1
                    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

                student_answers[q + 1] = self.choices[bubbled[1]] if bubbled[1] < len(self.choices) else 'X'

            total_questions = min(len(self.answer_key), q + 1)
            score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

            return {
                "correct": correct_answers,
                "total": total_questions,
                "percentage": round(score_percentage, 2),
                "answers": student_answers,
                "processed_image": self.image_to_base64(paper)
            }

        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}

    def image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def set_config(self, num_questions, choices, answer_key):
        self.num_questions = num_questions
        self.choices = choices
        self.answer_key = [choices.index(ans) if ans in choices else 0 for ans in answer_key]


omr_generator = OMRGenerator()
omr_scanner = OMRScanner()


@app.route('/')
def home():
    return "OMR Generator and Scanner API is running."


@app.route('/generate_omr', methods=['POST'])
def generate_omr():
    data = request.get_json()
    title = data.get('title', 'EXAM')
    num_questions = data.get('num_questions', 20)
    if num_questions > 30:
        return {"error": "Too many questions"}, 400
    choices = data.get('choices', ['A', 'B', 'C', 'D'])
    student_info = data.get('student_info', True)

    pdf_buffer = omr_generator.create_omr_sheet(num_questions, choices, student_info, title)
    return send_file(pdf_buffer, mimetype='application/pdf', as_attachment=True,
                     download_name=f'OMR_Sheet_{title.replace(" ", "_")}.pdf')


@app.route('/set_scanner_config', methods=['POST'])
def set_scanner_config():
    data = request.get_json()
    omr_scanner.set_config(
        data.get('num_questions', 20),
        data.get('choices', ['A', 'B', 'C', 'D', 'E']),
        data.get('answer_key', [])
    )
    return jsonify({"success": True})


@app.route('/process_omr', methods=['POST'])
def process_omr():
    data = request.get_json()
    image_data = data.get('image', '')

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]

    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = omr_scanner.process_omr_sheet(image)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
