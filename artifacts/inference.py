import cleantext
import torch
import transformers
import os
import joblib


class Inference:
    def __init__(self, model_path: str):
        """
        Initializes the Inference for sentiment analysis by loading a pre-trained model and tokenizer.

        Args:
            model_path (str): The file path to the directory containing the model and tokenizer.
        """
        self.device = torch.device("cpu")

        self.model = joblib.load(os.path.join(model_path, "sentiment_model.joblib"))
        self.tokenizer = joblib.load(
            os.path.join(model_path, "sentiment_tokenizer.joblib")
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        """
        Predicts the sentiment of a given input text.

        Args:
            text (str): The input text to analyze for sentiment.

        Returns:
            str: The predicted sentiment label ('positive' or 'negative')
        """
        try:
            with torch.no_grad():
                inputs = self._preprocess(text)
                # Move inputs to the same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)

                # Handle both older and newer model output formats
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            prediction = self._postprocess(logits)
            return prediction
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return f"error: {str(e)}"

    def _postprocess(self, logits: torch.Tensor) -> str:
        predicted_label = logits.argmax(dim=-1).item()
        return "positive" if predicted_label == 1 else "negative"

    def _preprocess(
        self, text: str
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        text = self._clean_text(text)
        tokens = self._tokenize(text)
        return tokens

    def _tokenize(
        self, text: str
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenizer(text, return_tensors="pt")

    @staticmethod
    def _clean_text(text: str) -> str:
        return cleantext.clean(
            text,
            to_ascii=False,
            lower=True,
            no_emoji=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            replace_with_url=" ",
            replace_with_email=" ",
            replace_with_phone_number=" ",
        )
