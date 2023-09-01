import glob
import pickle
import numpy
import os
import json
import openai
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Bidirectional
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization as BatchNorm, Bidirectional
import random

import glob
import pickle
import numpy
import os
from music21 import converter, instrument, note, chord
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

base_path = ""  #enter your base path here

mood_categories = {
    'happy': ['FF3_Battle_(Piano).mid', 'FF8_Shuffle_or_boogie_pc.mid', 'FFVII_BATTLE.mid', 'Ff7-Cinco.mid', 'Ff7-Jenova_Absolute.mid', 'Ff7-One_Winged.mid', 'Gold_Silver_Rival_Battle.mid', 'Kingdom_Hearts_Dearly_Beloved.mid', 'Kingdom_Hearts_Traverse_Town.mid', 'OTD5YA.mid'],
    'sad': ['EyesOnMePiano.mid', 'FF3_Third_Phase_Final_(Piano).mid', 'FF6epitaph_piano.mid', 'FFIXQuMarshP.mid', 'FFX_-_Ending_Theme_(Piano_Version)_-_by_Angel_FF.mid', 'Final_Fantasy_7_-_Judgement_Day_Piano.mid', 'In_Zanarkand.mid', 'Suteki_Da_Ne_(Piano_Version).mid', 'ViviinAlexandria.mid', 'Zelda_Overworld.mid'],
    'angry': ['0fithos.mid', 'DOS.mid', 'Eternal_Harvest.mid', 'Ff4-BattleLust.mid', 'Fiend_Battle_(Piano).mid', 'Fierce_Battle_(Piano).mid', 'Finalfantasy6fanfarecomplete.mid', 'JENOVA.mid', 'Life_Stream.mid', 'Oppressed.mid'],
    'nostalgic': ['AT.mid', 'Cids.mid', 'FF4.mid', 'FFIII_Edgar_And_Sabin_Piano.mid', 'FFIX_Piano.mid', 'FFVII_BATTLE.mid', 'Final_Fantasy_Matouyas_Cave_Piano.mid', 'Rachel_Piano_tempofix.mid', 'Rydia_pc.mid', 'VincentPiano.mid'],
    'mysterious': ['8.mid', 'BlueStone_LastDungeon.mid', 'ff4-fight1.mid', 'Finalfantasy5gilgameshp.mid', 'HighwindTakestotheSkies.mid', 'Still_Alive-1.mid', 'bcm.mid', 'costadsol.mid', 'decisive.mid', 'gerudo.mid'],
    'romantic': ['ff4-town.mid', 'ff4pclov.mid', 'ff7themep.mid', 'ff8-lfp.mid', 'FFIX_Piano.mid', 'Ff7-Cinco.mid', 'ahead_on_our_way_piano.mid', 'balamb.mid', 'braska.mid', 'caitsith.mid'],
    'other': ['beethoven_opus10_1.mid', 'beethoven_opus10_2.mid', 'beethoven_opus10_3.mid', 'beethoven_opus22_1.mid', 'beethoven_opus22_2.mid', 'beethoven_opus22_3.mid', 'beethoven_opus22_4.mid', 'beethoven_opus90_1.mid', 'beethoven_opus90_2.mid', 'islamei.mid']
}



def get_mood_from_text(text):
    openai.api_key = "" #your opeanai api here
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
             {"role": "system", "content": 'You are a helpful assistant that converts text to mood values. An example response would be: {"happy": 0.75, "sad": 0, "angry": 0, "nostalgic": 0, "mysterious": 0, "romantic": 0.25}'},
            {"role": "user", "content": f"Convert the following text to mood values in JSON format: {text}. The mood values should be for 'happy', 'sad', 'angry', 'nostalgic', 'mysterious', and 'romantic'. The values should add up to 1."}
        ]
    )
    mood_text = response['choices'][0]['message']['content']
    print("Mood Text:", mood_text)
    mood_input = json.loads(mood_text)
    return mood_input

def train_network():
    text = input("Enter a sentence to set the mood: ")
    mood_input = get_mood_from_text(text)
    notes = get_notes(mood_input)
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    train(model, network_input, network_output)

def get_notes(mood_input):
    notes = []
    total_files = 20  # Total number of files you want to use for training
    midi_path = os.path.join(base_path, "midi_songs/*.mid")

    for mood, files in mood_categories.items():
        num_files_for_mood = int(total_files * mood_input.get(mood, 0))

        # Reuse files if not enough are available
        num_repeats = num_files_for_mood // len(files)
        remainder = num_files_for_mood % len(files)

        selected_files = files * num_repeats
        selected_files += random.sample(files, remainder)

        # If still not enough, borrow from 'other' category
        if len(selected_files) < num_files_for_mood:
            additional_files_needed = num_files_for_mood - len(selected_files)
            additional_files = random.sample(mood_categories['other'], additional_files_needed)
            selected_files += additional_files

        for file_name in selected_files:
            file = os.path.join(base_path, "midi_songs", file_name)
            midi = converter.parse(file)
            print(f"Parsing {file} for mood {mood}")

            notes_to_parse = None
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    notes_file_path = os.path.join(base_path, 'data/notes')
    with open(notes_file_path, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)
    return (network_input, network_output)

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))  # New LSTM layer
    model.add(Dropout(0.3))  # New Dropout layer
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model



def train(model, network_input, network_output):
    weights_path = os.path.join(base_path, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=200, batch_size=2048, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()

import subprocess

# Path to the predict.py script
predict_script_path = "/content/drive/MyDrive/Backup and Sync.app/Classical-Piano-Composer-master/predict.py"

# Run the script
subprocess.run(['python', predict_script_path])
