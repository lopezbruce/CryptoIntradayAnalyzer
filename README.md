# CryptoIntradayAnalyzer

CryptoIntradayAnalyzer is a Python-based project that utilizes OpenAI's GPT-3.5-turbo model and embeddings to analyze and answer questions about intraday data of cryptocurrencies, specifically Bitcoin (BTCUSD).

## Components

The project consists of two main scripts:

1. `btcusd_intraday_embedding.py`: This script converts the raw JSON data of BTCUSD intraday movements into embeddings. These embeddings are a numerical representation of the data that can be used for efficient similarity search. The embeddings are saved into a CSV file for later use.

2. `btcusd_intraday_query.py`: This script reads the CSV file containing the embeddings, and uses these embeddings to find the most relevant data to a given query. It then constructs a message containing this data and sends it to the GPT model to generate an answer.

## Usage

First, run `requirements.txt` to generate the embeddings CSV file:

```bash
pip install requirements.txt
```

Then, run `btcusd_intraday_embedding.py` to generate the embeddings CSV file:

```bash
python btcusd_intraday_embedding.py
```

Then, run `btcusd_intraday_query.py` to start the query interface:

```bash
python btcusd_intraday_query.py
```

## Requirements

- Python 3.6 or later
- OpenAI Python package
- pandas
- numpy

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
