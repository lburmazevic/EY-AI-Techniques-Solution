# Translating Italian University Strategies into Funding Opportunities

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


This project translates Italian university strategic plans into relevant funding opportunities using retrieval, explainability methods, and optional LLM-generated summaries.

It was developed for the Artificial Intelligence Techniques class at LUISS University, within the Bachelor in Artificial Intelligence and Management.

## Team Members

- [Luka Burmazevic](https://github.com/lburmazevic)
- Evelina Ristovska
- [Filippa Gronberg](https://github.com/gronbergfillipa)

## Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About the Project

The workflow links institutional strategy documents to funding calls by combining:

- text extraction and chunking,
- semantic retrieval over a ChromaDB collection,
- ranking and explainability outputs,
- optional local LLM summaries through Ollama.

The goal is to support stakeholders in quickly identifying calls that align with strategic priorities.

## Built With

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama (`qwen3:0.6b`)

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`
- Ollama installed locally (<https://ollama.com>)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lburmazevic/uni-strategy-funding-matcher.git
   cd uni-strategy-funding-matcher
   ```

1. Create and activate a virtual environment:

   Linux/macOS/WSL:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

1. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

1. Start Ollama and pull the model:

   ```bash
   ollama serve
   ```

   In another terminal:

   ```bash
   ollama pull qwen3:0.6b
   ```

## Usage

1. Run notebooks in order (top to bottom):

   - `ingest.ipynb`
   - `query.ipynb`

   This prepares the vector database and retrieval outputs.

1. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

1. Open the local URL shown in the terminal (usually `http://localhost:8501`).

Optional: sample strategic-plan PDFs are available in `docs/upload_examples/` for quick testing.

## Project Structure

- `app.py`: Streamlit app entry point for the interactive workflow.
- `pipeline_core_methods.py`: Core methods for cleaning, chunking, retrieval, ranking, and XAI.
- `llm_explainer.py`: Ollama-based summarization and explanation layer.
- `ingest.ipynb`: Ingestion notebook for funding call PDFs.
- `query.ipynb`: Notebook for retrieval, ranking, and explanation experiments.
- `docs/fundingcalls/`: Source funding call PDFs.
- `docs/strategicplans/`: Strategic plan PDFs used as query/demo inputs.
- `data/chroma/`: Persistent ChromaDB storage.
- `outputs/`: Saved app and phase outputs.

## Contributing

This repository is primarily maintained as an academic project, but suggestions and improvements are welcome.

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/your-feature`)
1. Commit your changes (`git commit -m "Add your feature"`)
1. Push your branch (`git push origin feature/your-feature`)
1. Open a pull request

## License

Distributed under the MIT License. See `LICENSE` for details.

## Contact

- Luka Burmazevic - [@lburmazevic](https://github.com/lburmazevic)
- Evelina Ristovska
- Filippa Gronberg - [@gronbergfilippa](https://github.com/gronbergfilippa)

Project Link: <https://github.com/lburmazevic/uni-strategy-funding-matcher>

## Acknowledgments

- Artificial Intelligence Techniques course, LUISS University
- Bachelor in Artificial Intelligence and Management
- Open-source ecosystem: Streamlit, ChromaDB, Sentence Transformers, and Ollama

[contributors-shield]: https://img.shields.io/github/contributors/lburmazevic/uni-strategy-funding-matcher.svg?style=for-the-badge
[contributors-url]: https://github.com/lburmazevic/uni-strategy-funding-matcher/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lburmazevic/uni-strategy-funding-matcher.svg?style=for-the-badge
[forks-url]: https://github.com/lburmazevic/uni-strategy-funding-matcher/network/members
[issues-shield]: https://img.shields.io/github/issues/lburmazevic/uni-strategy-funding-matcher.svg?style=for-the-badge
[issues-url]: https://github.com/lburmazevic/uni-strategy-funding-matcher/issues
[license-shield]: https://img.shields.io/github/license/lburmazevic/uni-strategy-funding-matcher.svg?style=for-the-badge
[license-url]: https://github.com/lburmazevic/uni-strategy-funding-matcher/blob/main/LICENSE
[stars-shield]: https://img.shields.io/github/stars/lburmazevic/uni-strategy-funding-matcher.svg?style=for-the-badge
[stars-url]: https://github.com/lburmazevic/uni-strategy-funding-matcher/stargazers
