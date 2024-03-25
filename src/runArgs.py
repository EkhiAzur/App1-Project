from dataclasses import dataclass, field

@dataclass
class runArgs:

    data_path: str = field(
        default="data",
        metadata={"help": "Path to the data folder"},
    )

    model_name: str = field(
        default="ixa-ehu/roberta-eus-euscrawl-large-cased",
        metadata={"help": "Model name"},
    )

    max_length: int = field(
        default=512,
        metadata={"help": "Max length of the input"},
    )

    contrastive: bool = field(
        default=False,
        metadata={"help": "Whether to use contrastive learning"},
    )

    contrastive_temp: float = field(
        default=0.2,
        metadata={"help": "Contrastive temperature"},
    )

    contrastive_lam: float = field(
        default=0.1,
        metadata={"help": "Contrastive lambda"},
    )

    pooling_strategy: str = field(
        default="cls",
        metadata={"help": "Pooling strategy"},
    )