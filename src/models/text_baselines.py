"""Traditional text baselines: preprocessing -> TF-IDF -> linear classifier.

Each baseline is one scikit-learn Pipeline:

    TextPreprocessor(mode)  ->  TfidfVectorizer  ->  LogisticRegression | LinearSVC

The pipeline takes RAW text (text_raw / app input) and applies the Task 6
preprocessing itself through src.preprocessing.preprocess(). The saved
artifact therefore contains everything needed at inference time, and the
app never needs separate preprocessing code.

TextPreprocessor has no learned state (its fit() does nothing), so it
cannot leak information. TF-IDF vocabulary/IDF and classifier weights are
learned only from what .fit() receives, which must be the TRAIN split.

Note: LinearSVC outputs decision scores (signed distances), NOT
probabilities. Do not present them as confidence probabilities.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import hinglish
from src.preprocessing.text import MODES, PreprocessingConfig, preprocess

CLASSIFIERS = ("logistic_regression", "linear_svm")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Stateless sklearn wrapper around src.preprocessing.preprocess().

    Stores the preprocessing version / rules version it was built with and
    refuses to run if the installed preprocessing code differs, so a model
    can never silently receive differently-processed text.
    """

    def __init__(self, mode: str = "raw", options: dict | None = None,
                 preprocessing_version: str | None = None, rules_version: str | None = None):
        self.mode = mode
        self.options = options
        self.preprocessing_version = preprocessing_version
        self.rules_version = rules_version

    def _config(self) -> PreprocessingConfig:
        # Built at call time: no machine-specific paths are stored in the artifact.
        cfg = PreprocessingConfig(**(self.options or {}))
        if self.preprocessing_version and cfg.version != self.preprocessing_version:
            raise RuntimeError(f"Model expects preprocessing v{self.preprocessing_version}, "
                               f"installed config is v{cfg.version}")
        if self.mode == "hinglish_normalized" and self.rules_version:
            installed = hinglish.load_rules(cfg.normalization_rules).version
            if installed != self.rules_version:
                raise RuntimeError(f"Model expects Hinglish rules v{self.rules_version}, installed v{installed}")
        return cfg

    def fit(self, X, y=None):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._config()   # validate early
        return self

    def transform(self, X):
        cfg = self._config()
        return [preprocess(t, self.mode, cfg) for t in X]


def preprocessing_options(config: dict) -> dict:
    """Preprocessing options from config.yaml (excluding the machine-specific rules path)."""
    tp = config["text_preprocessing"]
    keys = ("version", "lowercase", "url_handling", "url_placeholder", "mention_handling",
            "mention_placeholder", "hashtag_handling", "emoji_handling", "repeated_char_normalization",
            "remove_stopwords")
    return {k: tp[k] for k in keys if k in tp}


def build_pipeline(classifier: str, mode: str, config: dict, seed: int) -> Pipeline:
    """Create an unfitted baseline pipeline from config.yaml settings."""
    if classifier not in CLASSIFIERS:
        raise ValueError(f"classifier must be one of {CLASSIFIERS}")
    base = config["text_model"]["baselines"]
    tf = dict(base["tfidf"])
    tf["ngram_range"] = tuple(tf["ngram_range"])
    options = preprocessing_options(config)
    pre = TextPreprocessor(mode=mode, options=options, preprocessing_version=options["version"],
                           rules_version=hinglish.load_rules().version if mode == "hinglish_normalized" else None)
    if classifier == "logistic_regression":
        clf = LogisticRegression(random_state=seed, **base["logistic_regression"])
    else:
        clf = LinearSVC(random_state=seed, **base["linear_svm"])
    return Pipeline([("preprocess", pre), ("tfidf", TfidfVectorizer(**tf)), ("clf", clf)])


def decision_scores(pipeline: Pipeline, texts):
    """Signed scores for the positive class (LR log-odds or SVM margin). Not probabilities."""
    return pipeline.decision_function(texts)
