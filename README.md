# Label-Guided In-Context Learning for Named Entity Recognition
This repo provides code and data associated with the paper ["Label-Guided In-Context Learning for Named Entity Recognition"](https://arxiv.org/pdf/2505.23722).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bflashcp3f/deer.git
cd deer
```

### 2. Set Up Environment

Using Conda (recommended):
```bash
conda env create -f environment.yaml
conda activate deer
```

Using pip:
```bash
python -m venv deer_env
source deer_env/bin/activate
pip install -r requirements.txt
```

### 3. Set Up API Keys

Set your OPENAI_API_KEY (or TOGETHER_API_KEY) keys as environment variables

### 4. Install from Source

```bash
pip install -e .
```

## Data Preparation

Download the preprocessed datasets from [Google Drive](https://drive.google.com/file/d/1RDmYCJuO1Sw_YRipa3HGsv1wacqP4Pom/view?usp=sharing) and extract them to the appropriate directories:

```bash
# Download deer_data.tar.gz from Google Drive link above
# Then extract the data
tar -xzf deer_data.tar.gz

# The data should be organized as follows:
data/
├── ncbi/
├── conll03/
├── bc2gm/
├── ontonotes/
└── tweetner7/
```

Each dataset directory contains train, validation/dev, and test splits in JSONL format, along with pre-computed embeddings.

## Supported Datasets

- **NCBI**: Biomedical entity recognition
- **CoNLL-03**: Popular NER benchmark (Person, Location, Organization, Misc)
- **bc2gm**: Gene mention detection in biomedical text
- **OntoNotes**: 18 entity types across multiple domains
- **TweetNER7**: Social media NER with 7 entity types

## Example Scripts

See the `scripts/` directory for dataset-specific examples:
```bash
# Run in-context learning step of DEER on NCBI
bash scripts/ncbi/run_deer_icl.sh 8 openai text-embedding-3-small openai gpt-4o-mini-2024-07-18 64 1.0 1.0 0.01

# Run error reflection step of DEER on NCBI
bash scripts/ncbi/run_deer_er.sh 8 deer openai text-embedding-3-small openai gpt-4o-mini-2024-07-18 64 1.0 1.0 0.01 1 0.75 0.75 0.95
```

## Evaluation

Detailed evaluation results can be found in the Jupyter notebooks under the `notebooks/` directory:

## Citation

If you use DEER in your research, please cite:
```bibtex
@misc{bai2025labelguidedincontextlearningnamed,
      title={Label-Guided In-Context Learning for Named Entity Recognition}, 
      author={Fan Bai and Hamid Hassanzadeh and Ardavan Saeedi and Mark Dredze},
      year={2025},
      eprint={2505.23722},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23722}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and feedback, please open an issue on GitHub.