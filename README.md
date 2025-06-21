# 🦷 Tooth Decay Detection and Severity Analysis Using Deep Learning

A computer vision project that uses YOLOv8 to detect dental conditions like tooth, caries, and cavity from images. It evaluates severity, generates a GPT-style summary, and provides a downloadable PDF report—all through an easy-to-use Streamlit interface.

---

## 📌 Features

* ⚡ Real-time detection of **tooth**, **caries**, and **cavity**
* 📊 Severity classification (Tooth → Caries → Cavity)
* 📝 GPT-style natural language summary for each case
* 📄 PDF report generation with timestamp, image, and result
* 🌐 Streamlit-based user interface (local or cloud)
* 💾 CSV logging of detection history

---

## 🔧 Technologies Used

* **Python 3.11+**
* **YOLOv8 / Ultralytics**
* **OpenCV**
* **Streamlit**
* **FPDF (for PDF generation)**
* **Pandas (for CSV logging)**

---

## 📁 Project Structure

```
📦ToothDecayDetection
 ┣ 📂weights/
 ┃ ┗ best.pt
 ┣ 📂dataset/
 ┃ ┗ train, valid, test
 ┣ train_model.py                 
 ┣ severity_log.csv         
 ┗ README.md
---

## 🖼️ How It Works

1. Upload a dental image (X-ray or intraoral photo).
2. YOLOv8 detects relevant regions (tooth, caries, cavity).
3. Severity is calculated based on detection priority.
4. A GPT-style summary is generated automatically.
5. Download a complete PDF report with all results.

---

## ✅ Severity Logic

| Detected Labels        | Severity     |
| ---------------------- | ------------ |
| Contains "cavity"      | Cavity       |
| Else contains "caries" | Caries       |
| Only "tooth"           | Tooth        |
| None                   | No Detection |

---

## 📄 Sample Output

* **Severity**: Caries
* **Summary**: "Caries (initial decay) detected. Early treatment such as fluoride application or filling may help prevent cavity formation."

---

## 🧠 Future Enhancements

* GPT-4 API for live chatbot interaction
* Real-time webcam-based detection
* Mobile version using Streamlit Cloud or Gradio
* Pediatric dental mode

