"""Dataset adapters: where each dataset lives and which columns mean what.

Column names below were VERIFIED by reading the actual file header
(Task 3). Label meanings come from the dataset card (Task 4) and are
kept in `documented`, separate from anything measured.

Add a new DatasetSpec when another dataset is placed in data/raw/.
"""

from dataclasses import dataclass, field

from src.utils.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: str                  # relative to data/raw
    annotation_file: str            # relative to the dataset directory
    id_column: str
    image_directory: str | None = field(default=None, kw_only=True)  # relative to the dataset directory
    text_column: str | None
    image_column: str | None
    primary_label: str
    secondary_labels: list[str] = field(default_factory=list)
    positive_value: str | None = None
    # Documented facts (from the paper/source), kept separate from measurements.
    documented: dict = field(default_factory=dict)
    # Task 3 recommendation (analyst judgement, see docs/DATASET_REPORT.md).
    recommended_role: str = "TO_BE_DECIDED"

    @property
    def root(self):
        return RAW_DIR / self.directory

    @property
    def annotation_path(self):
        return self.root / self.annotation_file

    @property
    def image_root(self):
        return self.root / self.image_directory if self.image_directory else self.root


MIMIC = DatasetSpec(
    name="MIMIC",
    directory="mimic",
    annotation_file="MIMIC2024.csv",
    id_column="FileName",           # also the image reference
    text_column="ExtractedText",
    image_column="FileName",
    primary_label="Misogyny",
    secondary_labels=["Objectification", "Prejudice", "Humiliation"],
    positive_value="1",
    documented={
        "full_name": "MIMIC: Misogyny Identification in Multimodal Internet "
                     "Content in Hindi-English Code-Mixed Language",
        "citation": "Singh, A., Sharma, D., & Singh, V. K. (2024). ACM Transactions "
                    "on Asian and Low-Resource Language Information Processing. "
                    "https://doi.org/10.1145/3656169",
        "tasks": "Binary misogyny identification; multi-label categorisation of "
                 "misogynistic memes (Objectification, Prejudice, Humiliation)",
        "task_scope": "misogyny specifically (not general offensiveness, hate, "
                      "abuse, sarcasm or sentiment)",
        "language": "Hindi-English code-mixed",
        "modalities": "meme image + text",
        "reported_split": "No official split in the release (dataset card shows a "
                          "single 5,054-row set). A third-party paper ('M3Hop-CoT', "
                          "arXiv:2410.09220, Table 1) used 4,044 train / 1,010 test; "
                          "that split is not present in the local file",
        "dataset_card": "https://huggingface.co/datasets/Aakash941/MIMIC-Meme-Dataset",
        "license": "CC-BY-4.0 (dataset card licence tag; confirm it matches your download source)",
        "label_value_meaning": "Misogyny: 1 = Misogynistic, 0 = Non Misogynistic (dataset card)",
        "text_origin": "OCR: the dataset card says text was extracted with 'EasyORC' "
                       "(presumably EasyOCR)",
    },
    recommended_role="PRIMARY TRAINING DATASET",
    image_directory="Files",
)

DATASETS = {"mimic": MIMIC}
