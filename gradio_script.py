{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d6b3bd1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import gradio as gr\n",
    "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1f91ccf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import gradio as gr\n",
    "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification\n",
    "\n",
    "# Load fine-tuned model and tokenizer\n",
    "model = DistilBertForSequenceClassification.from_pretrained('sentiment_model')\n",
    "tokenizer = DistilBertTokenizer.from_pretrained('sentiment_model')\n",
    "\n",
    "# Define prediction function\n",
    "def predict_sentiment(text):\n",
    "    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')\n",
    "    outputs = model(**inputs)\n",
    "    predicted_class = torch.argmax(outputs.logits).item()\n",
    "    sentiment = {0: 'Negative', 1: 'Positive'}\n",
    "    return sentiment[predicted_class]\n",
    "\n",
    "# Create Gradio interface\n",
    "iface = gr.Interface(\n",
    "    fn=predict_sentiment,\n",
    "    inputs=gr.inputs.Textbox(),\n",
    "    outputs=gr.outputs.Textbox(label=\"Predicted Sentiment\"),\n",
    "    title=\"Sentiment Analysis Demo\",\n",
    "    description=\"Enter a movie review, and the model will predict its sentiment.\",\n",
    ")\n",
    "\n",
    "# Launch the interface\n",
    "iface.launch()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
