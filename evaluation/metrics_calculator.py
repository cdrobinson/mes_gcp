import textstat
import nltk
from nltk.tokenize import word_tokenize # sent_tokenize not used currently
from typing import Dict, Any, List, Optional

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

class MetricsCalculator:
    """Calculates various ground truth-independent metrics for text."""

    def _get_words(self, text: str) -> List[str]:
        """Tokenizes text into words."""
        if not text:
            return []
        return word_tokenize(text.lower())

    def calculate_transcription_metrics(self,
                                        transcript: str,
                                        audio_duration_seconds: Optional[float] = None,
                                        stt_processing_time_seconds: Optional[float] = None
                                        ) -> Dict[str, Any]:
        """
        Calculates metrics for a given transcript.
        These metrics are relevant for evaluating the output of a Speech-to-Text system.
        """
        if not transcript:
            return {
                "transcript_length_chars": 0,
                "transcript_length_words": 0,
                "transcript_vocab_size": 0,
                "audio_duration_seconds": audio_duration_seconds,
                "words_per_minute": 0,
                "stt_processing_time_seconds": stt_processing_time_seconds
            }

        words = self._get_words(transcript)
        num_words = len(words)
        num_chars = len(transcript)
        vocab_size = len(set(words))
        wpm = 0

        if audio_duration_seconds and audio_duration_seconds > 0:
            minutes = audio_duration_seconds / 60
            wpm = round(num_words / minutes, 2) if minutes > 0 else 0

        return {
            "transcript_length_chars": num_chars,
            "transcript_length_words": num_words,
            "transcript_vocab_size": vocab_size,
            "audio_duration_seconds": audio_duration_seconds,
            "words_per_minute": wpm,
            "stt_processing_time_seconds": stt_processing_time_seconds
        }

    def calculate_llm_text_metrics(self,
                                   llm_output: str,
                                   llm_processing_time_seconds: Optional[float] = None,
                                   llm_metadata: Optional[Dict[str, Any]] = None
                                   ) -> Dict[str, Any]:
        """
        Calculates general text metrics for the output of an LLM.
        These metrics are broadly applicable for tasks like summarization, analysis, Q&A, etc.
        """
        if not llm_output:
            return {
                "llm_output_length_chars": 0,
                "llm_output_length_words": 0,
                "llm_output_vocab_size": 0,
                "readability_flesch_reading_ease": None,
                "readability_gunning_fog": None,
                "readability_dale_chall": None,
                "llm_processing_time_seconds": llm_processing_time_seconds,
                "llm_input_tokens": llm_metadata.get("prompt_token_count", 0) if llm_metadata else 0,
                "llm_output_tokens": llm_metadata.get("candidates_token_count", 0) if llm_metadata else 0,
                "llm_total_tokens": llm_metadata.get("total_token_count", 0) if llm_metadata else 0,
            }

        words = self._get_words(llm_output)
        num_words = len(words)
        num_chars = len(llm_output)
        vocab_size = len(set(words))

        try: flesch_ease = round(textstat.flesch_reading_ease(llm_output), 2)
        except: flesch_ease = None
        try: gunning_fog = round(textstat.gunning_fog(llm_output), 2)
        except: gunning_fog = None
        try: dale_chall = round(textstat.dale_chall_readability_score(llm_output), 2)
        except: dale_chall = None

        llm_meta = llm_metadata or {}
        return {
            "llm_output_length_chars": num_chars,
            "llm_output_length_words": num_words,
            "llm_output_vocab_size": vocab_size,
            "readability_flesch_reading_ease": flesch_ease,
            "readability_gunning_fog": gunning_fog,
            "readability_dale_chall": dale_chall,
            "llm_processing_time_seconds": llm_processing_time_seconds,
            "llm_input_tokens": llm_meta.get("prompt_token_count", 0),
            "llm_output_tokens": llm_meta.get("candidates_token_count", 0),
            "llm_total_tokens": llm_meta.get("total_token_count", 0),
        }

    def get_all_metrics(self,
                        transcript: str,
                        llm_output: str,
                        audio_duration_seconds: Optional[float] = None,
                        stt_processing_time_seconds: Optional[float] = None,
                        llm_processing_time_seconds: Optional[float] = None,
                        llm_metadata: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
        """
        Calculates and combines all relevant metrics.
        """
        metrics = {}
        trans_metrics = self.calculate_transcription_metrics(transcript, audio_duration_seconds, stt_processing_time_seconds)
        llm_metrics = self.calculate_llm_text_metrics(llm_output, llm_processing_time_seconds, llm_metadata)

        # Prefix to avoid name collisions and for clarity
        for k, v in trans_metrics.items():
            metrics[f"metric_transcript_{k}"] = v
        for k, v in llm_metrics.items():
            metrics[f"metric_llm_{k}"] = v
        return metrics


if __name__ == '__main__':
    calculator = MetricsCalculator()
    sample_transcript = "This is a sample transcript. It has several words and characters. This is a test."
    sample_llm_output = "This is a summary. It is concise and to the point. This is an LLM output."
    llm_meta_example = {
        "prompt_token_count": 50,
        "candidates_token_count": 25,
        "total_token_count": 75
    }

    all_m = calculator.get_all_metrics(
        transcript=sample_transcript,
        llm_output=sample_llm_output,
        audio_duration_seconds=10.5,
        stt_processing_time_seconds=2.1,
        llm_processing_time_seconds=5.3,
        llm_metadata=llm_meta_example
    )
    print("All Combined Metrics:", all_m)

    # Test with short text for readability
    short_text_llm = "Hi."
    short_text_transcript = "Hello there."
    all_m_short = calculator.get_all_metrics(short_text_transcript, short_text_llm, 5.0, 0.5, 0.2)
    print("All Combined Metrics (Short Text):", all_m_short)

    # Test with empty transcript/output
    empty_m = calculator.get_all_metrics("", "", 0, 0.1, 0.2)
    print("All Combined Metrics (Empty Text):", empty_m)