
# From Language to Melody: Music Generation

## Overview
This project, **From Language to Melody**, explores the connection between natural language and music by transforming text into musical compositions using AI-driven methods. By leveraging Natural Language Processing (NLP) models, including GPT-4, and Long Short-Term Memory (LSTM) networks, this project generates MIDI music based on user-provided text input. The system bridges literature and music by allowing any form of natural language input to be converted into structured MIDI files.

## Key Features
- **Text-to-Music Conversion**: Utilizes NLP techniques to convert natural language text into structured parameters that guide music generation.
- **MIDI Generation**: Produces MIDI music files with a single instrument based on mood and tone extracted from the text.
- **LSTM-Based Music Generation**: Trains neural networks using labeled MIDI data to generate compositions based on input mood values.

## Requirements
- **Python 3.x**
- **Dependencies**: Install the following packages using pip:
  ```bash
  pip install music21 keras tensorflow h5py openai
  ```

## Methodology
The project is divided into two main components:
1. **Text-to-Music Pipeline**: The input text is processed by an NLP model (GPT-4), which converts the text into a mood-based JSON structure, such as `{"happy": 0.75, "sad": 0, "nostalgic": 0.25}`. This structure is then passed to the music generation model.
2. **LSTM Music Generation**: The LSTM model takes the mood input and generates a MIDI file by predicting the sequence of notes. The model is trained on MIDI data labeled by mood, pitch, and duration.

## Training the Model
To train the model, use the following command:
```bash
python lstm.py
```
The script will load the MIDI files, preprocess the data, and train the neural network to generate music based on mood parameters. You can interrupt the training at any point, and the weights from the latest epoch will be saved.

## Generating Music
After training, generate music using the `predict.py` script:
```bash
python predict.py
```
The script will generate a MIDI file based on a given text input, using the saved model weights.

## Results
The project successfully generates music from text input, aligning the musical output with the mood and tone of the provided text. The final model produces MIDI compositions that reflect the emotional attributes of the input, creating a unique intersection between language and music.

## Future Work
This project is an initial exploration of text-to-music conversion. Future improvements may include:
- Expanding the dataset to improve model generalization.
- Experimenting with advanced models like GANs for more diverse music generation.
- Allowing for multi-instrumental compositions and integrating more complex music theory concepts.

## Credits
This project builds upon [Skuldur's Classical Piano Composer](https://github.com/Skuldur/Classical-Piano-Composer) and various music generation tutorials, adapting and expanding them to integrate text-based inputs using advanced NLP methods.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
