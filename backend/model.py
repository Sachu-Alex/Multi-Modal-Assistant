# Model Loading and Initialization
# This section handles loading of both vision and text models with fallback handling

try:
    import torch
    from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline

    # Determine the best available device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # Load BLIP model for Visual Question Answering
    model_name = "Salesforce/blip-vqa-base"
    print(f"⏳ Loading {model_name}...")

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()  # Set to evaluation mode for inference
    print(f"✅ Model {model_name} loaded on {device}")

    # Load DistilBERT model for Text Question Answering
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=0 if device == "cuda" else -1
    )
    print("✅ Text QA model loaded successfully")
    MODELS_LOADED = True

except Exception as e:
    # Fallback when models fail to load (e.g., missing dependencies)
    print(f"❌ Model loading failed: {e}")
    processor = model = qa_pipeline = None
    device = "cpu"
    MODELS_LOADED = False

def get_fallback_text_response(context, question):
    """Provide a helpful fallback response when text QA models aren't available"""
    if not context or not question:
        return "❌ Please provide both text and a question."
    
    return f"🤖 I can see your text ({len(context)} characters), but the AI models aren't loaded. Please check your installation."


def get_fallback_image_response(image, question):
    """Provide a helpful fallback response when image analysis models aren't available"""
    if not image:
        return "❌ Please upload an image."
    
    try:
        # Extract basic image information even without AI models
        width, height = image.size
        return f"🤖 I can see your image ({width}x{height} pixels), but the AI models aren't loaded. Please check your installation."
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"


def get_text_response(context, question):
    """Process text-based questions using the DistilBERT QA model"""
    if not MODELS_LOADED or not qa_pipeline:
        return get_fallback_text_response(context, question)
    
    try:
        # Use the text QA pipeline to find answers in the provided context
        result = qa_pipeline(question=question, context=context)
        
        # Check confidence score - if too low, indicate uncertainty
        if result['score'] < 0.1:
            return "🤖 I'm not sure about the answer based on the given text."
        return result['answer']
    except Exception as e:
        return f"❌ Error processing text: {str(e)}"


def get_model_response(image, question, context=None):
    """Main function to handle both image and text-based questions
    
    Args:
        image: PIL Image object for visual questions (optional)
        question: User's question string
        context: Text context for document-based questions (optional)
    
    Returns:
        str: AI-generated response or error message
    """
    # Handle case where models failed to load
    if not MODELS_LOADED:
        if context:
            return get_fallback_text_response(context, question)
        elif image:
            return get_fallback_image_response(image, question)
        else:
            return "❌ AI models are not loaded. Please check your installation."
    
    try:
        # Route to text processing if context is provided
        if context:
            return get_text_response(context, question)
            
        # Validate inputs for image processing
        if not question:
            return "❗ Please ask a question."

        if not image:
            return "❗ Please upload an image to ask questions about it."
            
        # Ensure question ends with question mark for better model performance
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
            
        # Process image and question through BLIP model
        inputs = processor(image, question, return_tensors="pt").to(device)
        
        # Generate response without gradient computation (inference mode)
        with torch.no_grad():
            output = model.generate(**inputs)
        
        # Decode the generated response
        answer = processor.decode(output[0], skip_special_tokens=True).strip()
        
        # Handle empty or uncertain responses
        if not answer or answer.lower() in ["", "i don't know", "i'm not sure"]:
            return "🤖 I'm not sure about the answer. Could you try asking in a different way?"
            
        # Capitalize first letter of response for better presentation
        return answer[0].upper() + answer[1:] if answer else "🤖 No answer generated."
            
    except Exception as e:
        return f"❌ Error: {str(e)}"