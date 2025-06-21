import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import pandas as pd
import os
from datetime import datetime
from fpdf import FPDF


model = YOLO("C:/Users/Ashwin A K/Desktop/Summer intern/toot_yolo_train/weights/best.pt")


st.set_page_config(page_title="Tooth Detection", layout="centered")
st.title("Tooth Detection with Severity, Summary & PDF")

def generate_summary(severity):
    if severity == "Cavity":
        return ("A cavity has been detected. This suggests advanced tooth decay "
                "that requires immediate dental attention to avoid complications.")
    elif severity == "Caries":
        return ("Caries (initial decay) detected. Early treatment such as fluoride "
                "application or filling may help prevent cavity formation.")
    elif severity == "Tooth":
        return ("No signs of decay detected. Maintain proper oral hygiene with regular "
                "brushing, flossing, and dental checkups.")
    else:
        return ("No objects detected. Please ensure the image is clear and properly focused. "
                "Try uploading a new image or consult a dental professional.")

def generate_pdf(severity, summary_text, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Tooth Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Severity Level: {severity}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Summary:\n{summary_text}")
    if image_path:
        pdf.image(image_path, x=10, y=None, w=180)
    file_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(file_name)
    return file_name


log_file = "severity_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Timestamp", "Severity"]).to_csv(log_file, index=False)

uploaded_file = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    results = model.predict(image_path, conf=0.25)
    annotated_img = results[0].plot()
    img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Extract class names
    class_names = [results[0].names[int(cls.item())].lower() for cls in results[0].boxes.cls]

    # Severity logic
    if "cavity" in class_names:
        severity = "Cavity"
    elif "caries" in class_names:
        severity = "Caries"
    elif all(name == "tooth" for name in class_names) and class_names:
        severity = "Tooth"
    else:
        severity = "No Detection"

    # Generate summary
    summary_text = generate_summary(severity)

    # Display results
    st.image(img_rgb, caption="Detected Image", use_column_width=True)
    st.markdown(f"Severity Level: *{severity}*")
    st.markdown("GPT-style Dental Summary:")
    st.write(summary_text)

    # Save image
    img_save_path = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(img_save_path, annotated_img)

    # Generate and download PDF
    pdf_file = generate_pdf(severity, summary_text, img_save_path)
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF Report", data=f, file_name=pdf_file, mime="application/pdf")

    # Log result
    log_df = pd.read_csv(log_file)
    log_df.loc[len(log_df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), severity]
    log_df.to_csv(log_file, index=False)