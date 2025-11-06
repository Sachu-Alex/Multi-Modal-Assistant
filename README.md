# 🧠 Multi-Modal Assistant

A powerful Streamlit-based application that combines image analysis and text processing capabilities using state-of-the-art AI models. Ask questions about images or analyze text documents with intelligent responses.

> 📖 **For complete documentation including problem statement, solution architecture, and technical details, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

## ✨ Features

- **🖼️ Image Analysis**: Upload images and ask questions about their content using BLIP (Bootstrapping Language-Image Pre-training) model
- **📝 Text Processing**: Analyze text documents and get answers to your questions using DistilBERT
- **💬 Interactive Chat**: Maintain conversation history with context-aware responses
- **🎯 Dual Modes**: Switch between image+text and text-only analysis modes
- **⚡ GPU Support**: Automatic CUDA detection for faster processing when available
- **🛡️ Fallback Mode**: Basic functionality even when PyTorch installation has issues

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multi_modal_assistant
   ```

2. **Install PyTorch (Windows-specific)**
   
   **For CPU-only (recommended for most users):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   
   **For CUDA 11.8 (if you have NVIDIA GPU with CUDA 11.8):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **For CUDA 12.1 (if you have NVIDIA GPU with CUDA 12.1):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 How to Use

### Image Analysis Mode
1. Select "Image + Text" mode from the sidebar
2. Upload an image (JPG, JPEG, or PNG format)
3. Ask questions about the image content
4. Get intelligent responses about what's in the image

### Text Analysis Mode
1. Select "Text Only" mode from the sidebar
2. Paste your text content (articles, documents, etc.)
3. Ask questions about the text
4. Receive accurate answers based on the content

## 🛠️ Technical Details

### Models Used
- **BLIP VQA**: `Salesforce/blip-vqa-base` for image question answering
- **DistilBERT**: `distilbert-base-cased-distilled-squad` for text question answering

### Architecture
```
multi_modal_assistant/
├── app.py                 # Main Streamlit application
├── backend/
│   ├── model.py          # AI model loading and inference
│   └── utils.py          # Utility functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Key Components
- **Model Management**: Automatic device detection (CPU/GPU)
- **Session State**: Maintains conversation history and context
- **Error Handling**: Graceful fallbacks for model failures
- **Responsive UI**: Clean, intuitive interface with emojis

## 📋 Requirements

- `streamlit>=1.25.0` - Web application framework
- `transformers>=4.31.0,<5.0.0` - Hugging Face transformers library
- `torch>=1.13.0,<2.3.0` - PyTorch for deep learning
- `Pillow>=9.0.0` - Image processing
- `sentence-transformers>=2.2.2` - Text embeddings
- `accelerate>=0.20.0` - Model optimization
- `torchvision>=0.14.0` - Computer vision utilities

## 🔧 Configuration

The application automatically detects available hardware:
- **GPU**: Uses CUDA if available for faster processing
- **CPU**: Falls back to CPU processing if GPU is not available

## 🎨 Features in Detail

### Image Analysis
- Supports JPG, JPEG, and PNG formats
- Real-time image preview
- Context-aware question answering
- Handles various image types and content

### Text Processing
- Large text document support
- Context-aware responses
- Confidence scoring for answers
- Fallback responses for low-confidence answers

### User Experience
- Clean, modern interface
- Conversation history
- Loading indicators
- Error handling with user-friendly messages

## 🐛 Troubleshooting

### Common Issues

1. **PyTorch DLL Import Error (Windows)**
   ```
   ImportError: DLL load failed while importing _C: The specified module could not be found.
   ```
   **Solution:**
   - **Step 1**: Download and install Microsoft Visual C++ Redistributable:
     - Go to: https://aka.ms/vs/17/release/vc_redist.x64.exe
     - Download and run the installer
     - Restart your computer after installation
   
   - **Step 2**: Reinstall PyTorch:
     ```bash
     pip uninstall torch torchvision torchaudio -y
     pip install torch torchvision torchaudio
     ```
   
   - **Alternative**: If the above doesn't work, try conda installation:
     ```bash
     conda install pytorch torchvision torchaudio cpuonly -c pytorch
     ```
   
   - **Fallback Mode**: If PyTorch still doesn't work, the app will use fallback functionality:
     - Basic image analysis (size, color mode)
     - Simple text processing with keyword matching
     - Limited but functional responses

2. **Model Loading Errors**
   - Ensure you have sufficient RAM (8GB+ recommended)
   - Check internet connection for model downloads
   - Verify CUDA installation if using GPU

3. **Memory Issues**
   - Close other applications to free up RAM
   - Consider using CPU-only mode for lower memory usage

4. **Slow Performance**
   - Ensure CUDA is properly installed for GPU acceleration
   - Check GPU memory availability

5. **Transformers Import Errors**
   - Make sure PyTorch is installed correctly first
   - Try reinstalling transformers: `pip uninstall transformers && pip install transformers`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the transformer models
- [Salesforce Research](https://github.com/salesforce/BLIP) for the BLIP model
- [Streamlit](https://streamlit.io/) for the web framework

---

**Made with ❤️ for intelligent multi-modal interactions**
