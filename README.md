# Map Captioning Pipeline

This repository contains a pipeline developed for captioning OpenStreetMap (OSM) map tiles. The pipeline generates descriptive captions for map tiles and creates a new dataset of map images paired with their corresponding captions. This dataset can be used for training and evaluating models in map interpretation tasks, geospatial analysis, and machine learning applications involving geographic data. The project aims to enhance the accessibility and usability of OSM data through automated caption generation.

## Key features:
- Efficient captioning of OSM map tiles.
- Generation of a comprehensive dataset with tile-caption pairs.
- Scalable pipeline for large-scale map data processing.

## Pipeline Overview:
The pipeline consists of two stages:

1. **Dataset downloading**: Fetching OSM map tiles dataset for captioning.
   - Configurable parameters:
     - `path`: Path to the Hugging Face dataset (default is `kraina/text2tile`).
     - `split`: Dataset split.
  
Here's the corrected section:


1. **Captioning**: Generating captions for the map tiles.
   - Configurable parameters:
     - `provider`: Name of the captioning provider. Possible options:
       - `openai`
       - `vertexai`
       - `ollama`
     - `model`: Name of the model that supports vision capabilities for captioning.
     - `prompt_template`: Name of the prompt template from the `data/prompt_templates` directory.
    
## Results Exploration:
The results from the pipeline, including the generated captions and dataset, are available for further exploration in the file `eda/results/exploration.ipynb`.

## Initialization Instructions:

1. **Set up environment variables:**
   - Copy the `.env.example` file to a new `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Update the `.env` file with your OpenAI and Google Cloud credentials:
     ```
     # openai access
     OPENAI_API_KEY="<your OpenAI API key>"

     # vertexai access
     GOOGLE_APPLICATION_CREDENTIALS="<path to your GCP service account JSON file>"
     ```

2. **Install dependencies**:
   - Use [PDM](https://pdm.fming.dev/) to install all required dependencies:
     ```bash
     pdm install
     ```

3. **Run the pipeline**:
   Before running the pipeline ensure that there are provided needed credentials.
   In case using the ollama server, ensure that the server is running and the selected model is available.
   
   - Use DVC to run the pipeline, with parameters configured in `params.yaml`:
     ```bash
     dvc repro
     ```


