<<<<<<< HEAD
# Cat & Dog classification with Resnet50 and SVM

This project uses the ResNet50 model for feature extraction and SVM (Support Vector Machine) for classification to distinguish between cat and dog images. The application is built with Flask and deployed on Render, offering a web interface where users can upload images and receive classification results.

## Project Structure
```
.../
├── data/                       
│   ├── training_set/            
│   ├── test_set/                 
(for future development)
│── train    
│   ├── model.pkl            
│   ├── ResNet.ipynb  
│   ├── resnet50_feature_extractor.pth
├── app.py                       
├── feature_extractor.py   
├── requirements.txt             
```


## 🚀 Getting Started

### Prerequisites

1. **Clone Repository**
   ```bash
   git clone <link>
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```


### 🎮 Running the Application in Localhost
```bash
flask run
```
Then open your browser to `http://localhost:8501`

![Streamlit UI](image\interface.png)
*Streamlit interface showing PDF viewer and chat functionality*


## 💡 Usage Tips

1. **Upload PDF**: Use the file uploader in the Streamlit interface
2. **Ask Questions**: Start chatting with your PDF through the chat interface"# Ollama_PDF_Chatbot" 


