import os
import pandas as pd

def count_anns(annotation_path):
    header = ["start", "mark", "tempo", "key", "instruments", "generator"]
    tot_count = 0
    has_melody = 0
    all_melody = 0
    same_instr_ct = 0
    songs_w_1_instr_mel = []

    for num in range(1, 3001):
        tot_count += 1
        format_num = f"{num:04d}"

        path = f"{annotation_path}/{format_num}_segments.arff"
        df = pd.read_csv(path, comment="@", header=None, quotechar="'")
        df.columns = header

        instruments_unprocessed = df['instruments'].tolist()
        instruments_unprocessed.pop()
        instruments = [inst.replace("'", "").replace("[", "").replace("]", "").split(",") for inst in instruments_unprocessed]

        generator_unprocessed = df['generator'].tolist()
        generator_unprocessed.pop()
        generator = [gen.replace("'", "").replace("[", "").replace("]", "").split(",") for gen in generator_unprocessed]

        if num == 1:
            print(instruments)
            print(generator)
        
        mel = False
        all_mel = 0
        same_instr = True
        instr_name = ""

        for i in range(len(instruments)):
            if generator[i][0] == "MelodyBow":
                mel = True
                all_mel += 1
                if i == 0:
                    instr_name = instruments[i][0]
                elif instruments[i][0] != instr_name:
                    same_instr = False
        
        if mel:
            has_melody += 1
        if all_mel == len(instruments):
            all_melody += 1
            if same_instr:
                same_instr_ct += 1
                songs_w_1_instr_mel.append((format_num, instr_name))

    # save songs_w_1_instr_mel to a csv file
    df_songs = pd.DataFrame(songs_w_1_instr_mel, columns=["song_number", "instrument"])
    df_songs.to_csv("songs_w_1_instr_mel.csv", index=False)

    return tot_count, has_melody, all_melody, same_instr_ct

def keep_full_mels():
    df = pd.read_csv("songs_w_1_instr_mel.csv")
    file_set = set()
    for _, row in df.iterrows():
        file_name = f"{int(row['song_number']):04d}_{row['instrument']}.flac"
        file_set.add(file_name)
        print(file_name)

    tracks_path = "dataset/tracks"
    for file in os.listdir(tracks_path):
        if file not in file_set:
            os.remove(os.path.join(tracks_path, file))
            print(f"Removed {file}")

#print(count_anns('dataset/annotations'))
keep_full_mels()